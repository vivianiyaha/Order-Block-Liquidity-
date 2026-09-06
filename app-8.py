"""
Smart Money Concepts (SMC) Trading Signal Generator
Forex & Bitcoin — Liquidity Sweep + Order Block Detection

Single-file Streamlit application.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="SMC Signal Generator | Forex & BTC",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# CUSTOM CSS — Blue / White / Green professional theme
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --navy: #0b2545;
        --blue: #1657c4;
        --light-blue: #eaf2ff;
        --green: #12965a;
        --light-green: #e6f8ee;
        --white: #ffffff;
    }

    .stApp {
        background-color: #f6f9fc;
    }

    h1, h2, h3, h4 {
        color: var(--navy) !important;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--navy);
    }
    section[data-testid="stSidebar"] * {
        color: #eaf2ff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #12305c;
        border-radius: 10px;
        border: 1px solid #2a4d7f;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, var(--blue), var(--navy));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 10px rgba(22, 87, 196, 0.25);
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(22, 87, 196, 0.35);
        color: white;
    }

    /* Cards */
    .signal-card {
        border-radius: 16px;
        padding: 22px 20px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(11, 37, 69, 0.08);
        border: 1px solid #dce7f7;
        height: 100%;
    }
    .card-navy { background-color: var(--light-blue); border-left: 6px solid var(--navy); }
    .card-green { background-color: var(--light-green); border-left: 6px solid var(--green); }
    .card-buy { background-color: var(--light-green); border-left: 6px solid var(--green); }
    .card-sell { background-color: #ffecec; border-left: 6px solid #c62828; }
    .card-neutral { background-color: #f0f2f6; border-left: 6px solid #8a94a6; }

    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #5a6b85;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--navy);
    }
    .card-value.green { color: var(--green); }
    .card-value.red { color: #c62828; }
    .card-value.blue { color: var(--blue); }

    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.04em;
    }
    .badge-buy { background-color: var(--green); color: white; }
    .badge-sell { background-color: #c62828; color: white; }
    .badge-neutral { background-color: #8a94a6; color: white; }

    .footer-note {
        color: #7a8699;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 30px;
    }

    hr {
        border-top: 1px solid #dce7f7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------
FOREX_TICKERS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]

TIMEFRAME_MAP = {
    "15m": {"interval": "15m", "period": "5d"},
    "1h": {"interval": "1h", "period": "1mo"},
    "4h": {"interval": "1h", "period": "3mo"},   # yfinance has no native 4h -> resample from 1h
    "1D": {"interval": "1d", "period": "1y"},
}

RISK_REWARD = 2.5          # target R:R multiple used for TP calculation
CONFIDENCE_DISPLAY = 75    # fixed display confidence per spec

# --------------------------------------------------------------------------------------
# DATA FETCHING
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, timeframe: str) -> pd.DataFrame:
    cfg = TIMEFRAME_MAP[timeframe]
    df = yf.download(
        tickers=ticker,
        period=cfg["period"],
        interval=cfg["interval"],
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten possible MultiIndex columns (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Resample to synthetic 4h bars from 1h data
    if timeframe == "4h":
        df = (
            df.resample("4h")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
        )

    return df


# --------------------------------------------------------------------------------------
# SWING POINT DETECTION
# --------------------------------------------------------------------------------------
def find_swing_points(df: pd.DataFrame, lookback: int = 3):
    """
    Fractal-style swing high/low detection.
    A swing high is a candle whose High is greater than `lookback` candles
    on either side; a swing low is the mirror condition.
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    swing_high_idx = []
    swing_low_idx = []

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback : i + lookback + 1]
        window_low = lows[i - lookback : i + lookback + 1]

        if highs[i] == window_high.max():
            swing_high_idx.append(i)
        if lows[i] == window_low.min():
            swing_low_idx.append(i)

    return swing_high_idx, swing_low_idx


# --------------------------------------------------------------------------------------
# CORE SMC STRATEGY: LIQUIDITY SWEEP + ORDER BLOCK
# --------------------------------------------------------------------------------------
def detect_signal(df: pd.DataFrame, lookback: int = 3, impulse_lookahead: int = 4):
    """
    Scans recent price action for a liquidity sweep of a prior swing point
    followed by an impulsive move back through structure, then locates the
    last opposite-colored candle before the impulse as the Order Block.

    Returns a dict describing the setup, or None if nothing qualifies.
    """
    if len(df) < (lookback * 2 + impulse_lookahead + 5):
        return None

    swing_high_idx, swing_low_idx = find_swing_points(df, lookback=lookback)
    n = len(df)
    closes = df["Close"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values

    best_setup = None

    # ---- Look for SELL setup: sweep of a swing HIGH, then bearish impulse ----
    for sh_i in reversed(swing_high_idx):
        swing_price = highs[sh_i]
        # search forward from the swing for a candle that sweeps above it
        for sweep_i in range(sh_i + 1, min(sh_i + 40, n)):
            if highs[sweep_i] > swing_price and closes[sweep_i] < swing_price:
                # liquidity sweep candle found (wicks above, closes back below)
                impulse_end = min(sweep_i + impulse_lookahead, n - 1)
                impulse_move = closes[impulse_end] - closes[sweep_i]
                impulse_range = df["High"].iloc[sweep_i:impulse_end + 1].max() - \
                                 df["Low"].iloc[sweep_i:impulse_end + 1].min()

                if impulse_move < 0 and impulse_range > 0:
                    # find last bullish (opposite-colored) candle before the impulse leg down
                    ob_idx = None
                    for j in range(sweep_i, max(sweep_i - 6, -1), -1):
                        if closes[j] > opens[j]:
                            ob_idx = j
                            break
                    if ob_idx is not None:
                        setup = {
                            "type": "SELL",
                            "swing_idx": sh_i,
                            "sweep_idx": sweep_i,
                            "ob_idx": ob_idx,
                            "ob_high": highs[ob_idx],
                            "ob_low": lows[ob_idx],
                        }
                        if best_setup is None or sweep_i > best_setup["sweep_idx"]:
                            best_setup = setup
                break  # only test first sweep after this swing

    # ---- Look for BUY setup: sweep of a swing LOW, then bullish impulse ----
    for sl_i in reversed(swing_low_idx):
        swing_price = lows[sl_i]
        for sweep_i in range(sl_i + 1, min(sl_i + 40, n)):
            if lows[sweep_i] < swing_price and closes[sweep_i] > swing_price:
                impulse_end = min(sweep_i + impulse_lookahead, n - 1)
                impulse_move = closes[impulse_end] - closes[sweep_i]
                impulse_range = df["High"].iloc[sweep_i:impulse_end + 1].max() - \
                                 df["Low"].iloc[sweep_i:impulse_end + 1].min()

                if impulse_move > 0 and impulse_range > 0:
                    ob_idx = None
                    for j in range(sweep_i, max(sweep_i - 6, -1), -1):
                        if closes[j] < opens[j]:
                            ob_idx = j
                            break
                    if ob_idx is not None:
                        setup = {
                            "type": "BUY",
                            "swing_idx": sl_i,
                            "sweep_idx": sweep_i,
                            "ob_idx": ob_idx,
                            "ob_high": highs[ob_idx],
                            "ob_low": lows[ob_idx],
                        }
                        if best_setup is None or sweep_i > best_setup["sweep_idx"]:
                            best_setup = setup
                break

    return best_setup


def build_trade_levels(df: pd.DataFrame, setup: dict):
    """Compute entry, stop-loss and take-profit from the identified order block."""
    ob_high = setup["ob_high"]
    ob_low = setup["ob_low"]
    ob_mid = (ob_high + ob_low) / 2

    if setup["type"] == "SELL":
        entry = ob_high
        stop_loss = ob_high * 1.0015 if ob_high else ob_high + (ob_high - ob_low) * 0.25
        stop_loss = max(stop_loss, ob_high + (ob_high - ob_low) * 0.15)
        risk = stop_loss - entry
        take_profit = entry - risk * RISK_REWARD
    else:  # BUY
        entry = ob_low
        stop_loss = ob_low * 0.9985 if ob_low else ob_low - (ob_high - ob_low) * 0.25
        stop_loss = min(stop_loss, ob_low - (ob_high - ob_low) * 0.15)
        risk = entry - stop_loss
        take_profit = entry + risk * RISK_REWARD

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk": abs(entry - stop_loss),
        "reward": abs(take_profit - entry),
    }


# --------------------------------------------------------------------------------------
# CHART BUILDER
# --------------------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, setup: dict, levels: dict, ticker: str, timeframe: str):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#12965a",
            decreasing_line_color="#c62828",
            increasing_fillcolor="#12965a",
            decreasing_fillcolor="#c62828",
            name="Price",
        ),
        row=1, col=1,
    )

    vol_colors = np.where(df["Close"] >= df["Open"], "#12965a", "#c62828")
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume", opacity=0.6),
        row=2, col=1,
    )

    if setup is not None:
        ob_idx = setup["ob_idx"]
        swing_idx = setup["swing_idx"]
        sweep_idx = setup["sweep_idx"]

        ob_time = df.index[ob_idx]
        end_time = df.index[min(sweep_idx + 6, len(df) - 1)]

        box_color = "rgba(198,40,40,0.18)" if setup["type"] == "SELL" else "rgba(18,150,90,0.18)"
        line_color = "#c62828" if setup["type"] == "SELL" else "#12965a"

        # Order Block zone
        fig.add_shape(
            type="rect",
            x0=ob_time, x1=end_time,
            y0=setup["ob_low"], y1=setup["ob_high"],
            fillcolor=box_color,
            line=dict(color=line_color, width=1.5),
            row=1, col=1,
        )
        fig.add_annotation(
            x=ob_time, y=setup["ob_high"],
            text="Order Block",
            showarrow=False,
            yshift=14,
            font=dict(color=line_color, size=11, family="Segoe UI"),
            row=1, col=1,
        )

        # Liquidity sweep marker ($$$)
        swing_price = df["High"].iloc[swing_idx] if setup["type"] == "SELL" else df["Low"].iloc[swing_idx]
        fig.add_annotation(
            x=df.index[swing_idx], y=swing_price,
            text="$$$",
            showarrow=True,
            arrowhead=2,
            ay=-30 if setup["type"] == "SELL" else 30,
            font=dict(color="#1657c4", size=13, family="Segoe UI Black"),
            row=1, col=1,
        )

        # Entry / SL / TP lines
        fig.add_hline(y=levels["entry"], line_dash="dot", line_color="#1657c4",
                       annotation_text="Entry", annotation_position="right", row=1, col=1)
        fig.add_hline(y=levels["stop_loss"], line_dash="dot", line_color="#c62828",
                       annotation_text="Stop Loss", annotation_position="right", row=1, col=1)
        fig.add_hline(y=levels["take_profit"], line_dash="dot", line_color="#12965a",
                       annotation_text="Take Profit", annotation_position="right", row=1, col=1)

    fig.update_layout(
        title=f"{ticker} — {timeframe} Chart (Smart Money Concepts)",
        template="plotly_white",
        height=650,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Segoe UI, sans-serif", color="#0b2545"),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def format_price(value: float, ticker: str) -> str:
    if "JPY" in ticker:
        return f"{value:,.3f}"
    if ticker in CRYPTO_TICKERS:
        return f"{value:,.2f}"
    return f"{value:,.5f}"


# --------------------------------------------------------------------------------------
# SIDEBAR — INPUTS
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Signal Settings")
    st.markdown("---")

    asset_class = st.selectbox("Asset Class", ["Forex", "Cryptocurrency"])

    if asset_class == "Forex":
        ticker = st.selectbox("Currency Pair", FOREX_TICKERS)
    else:
        ticker = st.selectbox("Crypto Pair", CRYPTO_TICKERS)

    timeframe = st.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()), index=2)

    st.markdown("---")
    generate = st.button("🚀 Generate Signal")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.78rem; color:#c8d6ee; line-height:1.5;">
        <b>Strategy Logic</b><br>
        1. Detect a liquidity sweep of a swing high/low ($$$)<br>
        2. Confirm an impulsive reversal move<br>
        3. Locate the last opposite-colored candle (Order Block)<br>
        4. Enter on retracement into the OB with a fixed R:R target
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# MAIN HEADER
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
        <div style="font-size:2rem;">📊</div>
        <div>
            <h1 style="margin-bottom:0;">SMC Signal Generator</h1>
            <div style="color:#5a6b85; font-size:0.95rem;">
                Liquidity Sweep &amp; Order Block Detection — Forex &amp; Bitcoin
            </div>
        </div>
    </div>
    <hr>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --------------------------------------------------------------------------------------
# GENERATE SIGNAL
# --------------------------------------------------------------------------------------
if generate:
    with st.spinner(f"Fetching {ticker} data and scanning market structure..."):
        df = fetch_data(ticker, timeframe)

        if df.empty or len(df) < 30:
            st.error(
                f"⚠️ Could not load sufficient data for **{ticker}** on the **{timeframe}** "
                "timeframe. Try a different pair or timeframe."
            )
            st.session_state.last_result = None
        else:
            setup = detect_signal(df)
            levels = build_trade_levels(df, setup) if setup else None
            st.session_state.last_result = {
                "df": df,
                "setup": setup,
                "levels": levels,
                "ticker": ticker,
                "timeframe": timeframe,
            }

# --------------------------------------------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------------------------------------------
result = st.session_state.last_result

if result is None:
    st.info("👈 Configure your asset and timeframe in the sidebar, then click **Generate Signal**.")
else:
    df = result["df"]
    setup = result["setup"]
    levels = result["levels"]
    r_ticker = result["ticker"]
    r_timeframe = result["timeframe"]

    # ---- Signal Panel ----
    st.markdown("### 🎯 Trade Setup")

    if setup is None:
        st.markdown(
            """
            <div class="signal-card card-neutral">
                <div class="card-title">Signal Status</div>
                <div class="card-value">No Qualifying Setup Found</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "No liquidity sweep + Order Block confluence was detected in the current data window. "
            "Try a different timeframe or asset."
        )
    else:
        signal_type = setup["type"]
        badge_class = "badge-buy" if signal_type == "BUY" else "badge-sell"
        card_class = "card-buy" if signal_type == "BUY" else "card-sell"
        value_color = "green" if signal_type == "BUY" else "red"

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(
                f"""
                <div class="signal-card {card_class}">
                    <div class="card-title">Signal</div>
                    <span class="badge {badge_class}">{signal_type}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="signal-card card-navy">
                    <div class="card-title">Confidence</div>
                    <div class="card-value blue">{CONFIDENCE_DISPLAY}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="signal-card card-navy">
                    <div class="card-title">Entry Price</div>
                    <div class="card-value">{format_price(levels['entry'], r_ticker)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="signal-card card-sell">
                    <div class="card-title">Stop Loss</div>
                    <div class="card-value red">{format_price(levels['stop_loss'], r_ticker)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                f"""
                <div class="signal-card card-buy">
                    <div class="card-title">Take Profit</div>
                    <div class="card-value green">{format_price(levels['take_profit'], r_ticker)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        rr_ratio = levels["reward"] / levels["risk"] if levels["risk"] else 0
        st.caption(
            f"Order Block range: **{format_price(setup['ob_low'], r_ticker)} – "
            f"{format_price(setup['ob_high'], r_ticker)}**  |  "
            f"Risk:Reward ≈ **1:{rr_ratio:.1f}**  |  "
            f"Liquidity swept at index **{df.index[setup['swing_idx']].strftime('%Y-%m-%d %H:%M')}**"
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- Chart ----
    st.markdown("### 📈 Market Structure Chart")
    fig = build_chart(df, setup, levels, r_ticker, r_timeframe)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="footer-note">
        This tool is for educational purposes only and does not constitute financial advice.
        Trading forex and cryptocurrency involves substantial risk of loss.
        </div>
        """,
        unsafe_allow_html=True,
    )
