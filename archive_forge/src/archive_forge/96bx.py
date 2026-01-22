import pandas as pd
import sqlite3
import math
import numpy as np
from typing import List, Dict, Any

# Constants
# The path to the database file
DB_PATH = "stock_data.db"

stock_data_frame: pd.DataFrame = pd.DataFrame(
    columns=[
        "price_relative_50_200",  # Price Relative 50/200
        "price_relative_20_50",  # Price Relative 20/50
        "price_relative_20_200",  # Price Relative 20/200
        "volume_relative_50",  # Volume Relative 50
        "volume_relative_200",  # Volume Relative 200
        "volume_obv_divergence",  # Volume OBV Divergence
        "on_balance_volume_ema_50",  # On Balance Volume EMA 50
        "chaikin_oscillator_3_10",  # Chaikin Oscillator
        "chaikin_volatility_10_10",  # Chaikin Volatility
        "volatility_atr_based_14",  # Volatility ATR Based
        "volatility_std_dev_based_20",  # Volatility STD Dev Based
        "volatility_rvi_14",  # Volatility RVI
        "volatility_rvi_std_dev_20",  # Volatility RVI STD Dev
        "cycle_ht_dcperiod",  # Cycle HT DC Period
        "cycle_ht_dcphase",  # Cycle HT DC Phase
        "cycle_ht_phasor_inphase",  # Cycle HT Phasor Inphase
        "cycle_ht_phasor_quadrature",  # Cycle HT Phasor Quadrature
        "cycle_ht_sine_sine",  # Cycle HT Sine Sine
        "cycle_ht_sine_leadsine",  # Cycle HT Sine Lead Sine
        "cycle_ht_trendmode",  # Cycle HT Trend Mode
        "cycle_ht_trendline",  # Cycle HT Trend Line
    ]
)


class StockDatabase:
    def __init__(self, db_path: str):
        """Initialize the database connection."""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.initialize_database()

    def execute_query(self, query: str, params: tuple = ()):
        """Execute a single SQL query."""
        self.cursor.execute(query, params)
        self.conn.commit()

    def initialize_database(self):
        """Create tables if they do not exist."""
        tables = {
            "Market_Data": """
                CREATE TABLE IF NOT EXISTS Market_Data (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER
                )
            """,
            "Adjusted_Market_Data": """
                CREATE TABLE IF NOT EXISTS Adjusted_Market_Data (
                    ticker TEXT,
                    adjusted_close REAL,
                    adjusted_volume INTEGER
                )
            """,
            "Dividends_Splits": """
                CREATE TABLE IF NOT EXISTS Dividends_Splits (
                    ticker TEXT,
                    dividends REAL,
                    stock_splits REAL
                )
            """,
            "Moving_Averages": """
                CREATE TABLE IF NOT EXISTS Moving_Averages (
                    ticker TEXT,
                    ma_5 REAL,  -- Moving Average 5
                    ma_15 REAL,  -- Moving Average 15
                    ma_50 REAL,  -- Moving Average 50
                    ma_200 REAL,  -- Moving Average 200
                    ma_365 REAL,  -- Moving Average 365
                    ema_5 REAL,  -- Exponential Moving Average 5
                    ema_15 REAL,  -- Exponential Moving Average 15
                    ema_50 REAL,  -- Exponential Moving Average 50
                    ema_200 REAL,  -- Exponential Moving Average 200
                    ema_365 REAL,  -- Exponential Moving Average 365
                    mma_5 REAL,  -- Multiface Moving Average 5
                    mma_15 REAL,  -- Multiface Moving Average 15
                    mma_50 REAL,  -- Multiface Moving Average 50
                    mma_200 REAL,  -- Multiface Moving Average 200
                    mma_365 REAL,  -- Multiface Moving Average 365
                    tema_5 REAL,  -- Triple Exponential Moving Average 5
                    tema_15 REAL,  -- Triple Exponential Moving Average 15
                    tema_50 REAL,  -- Triple Exponential Moving Average 50
                    tema_200 REAL,  -- Triple Exponential Moving Average 200
                    tema_365 REAL  -- Triple Exponential Moving Average 365
                )
            """,
            "Oscillators_Momentum": """
                CREATE TABLE IF NOT EXISTS Oscillators_Momentum (
                    ticker TEXT,
                    rsi REAL,  -- Relative Strength Index (RSI)
                    mfi REAL,  -- Money Flow Index (MFI)
                    tsi REAL,  -- True Strength Index (TSI)
                    uo REAL,  -- Ultimate Oscillator (UO)
                    stoch_rsi REAL,  -- Stochastic RSI
                    stoch_rsi_k REAL,  -- Stochastic RSI %K
                    stoch_rsi_d REAL,  -- Stochastic RSI %D
                    wr REAL,  -- Williams %R
                    ao REAL,  -- Awesome Oscillator
                    kama REAL,  -- Kaufman's Adaptive Moving Average (KAMA)
                    ppo REAL,  -- Percentage Price Oscillator (PPO)
                    ppo_signal REAL,  -- PPO Signal Line
                    ppo_hist REAL,  -- PPO Histogram
                    pvo REAL,  -- Percentage Volume Oscillator (PVO)
                    pvo_signal REAL,  -- PVO Signal Line
                    pvo_hist REAL,  -- PVO Histogram
                    roc REAL,  -- Rate of Change (ROC)
                    roc_100 REAL,  -- Rate of Change (ROC) 100
                    roc_100_sma REAL  -- Rate of Change (ROC) 100 Simple Moving Average (SMA)
                )
            """,
            "Volatility_Indicators": """
                CREATE TABLE IF NOT EXISTS Volatility_Indicators (
                    ticker TEXT,
                    bollinger_bands REAL,  -- Bollinger Bands
                    bollinger_bands_upper REAL,  -- Bollinger Bands Upper Band
                    bollinger_bands_lower REAL,  -- Bollinger Bands Lower Band
                    bollinger_bands_middle REAL,  -- Bollinger Bands Middle Band
                    keltner_channels REAL,  -- Keltner Channels
                    keltner_channels_upper REAL,  -- Keltner Channels Upper Band
                    keltner_channels_lower REAL,  -- Keltner Channels Lower Band
                    keltner_channels_middle REAL,  -- Keltner Channels Middle Band
                    donchian_channels REAL,  -- Donchian Channels
                    donchian_channels_upper REAL,  -- Donchian Channels Upper Band
                    donchian_channels_lower REAL,  -- Donchian Channels Lower Band
                    donchian_channels_middle REAL,  -- Donchian Channels Middle Band
                    atr REAL,  -- ATR(Average True Range)
                    true_range REAL,  -- True Range
                    natr REAL  -- NATR(Normalized Average True Range)
                )
            """,
            "Volume_Indicators": """
                CREATE TABLE IF NOT EXISTS Volume_Indicators (
                    ticker TEXT,
                    adi REAL,  -- ADI(Accumulation Distribution Index)
                    obv REAL,  -- OBV(On Balance Volume)
                    cmf REAL,  -- CMF(Chaikin Money Flow)
                    fi REAL,  -- FI(Force Index)
                    em REAL,  -- EM(Ease of Movement)
                    sma_em REAL,  -- SMA(Simple Moving Average) Ease of Movement
                    vpt REAL,  -- VPT(Volume Price Trend)
                    nvi REAL,  -- NVI(Negative Volume Index)
                    vwap REAL  -- VWAP(Volume Weighted Average Price)
                )
            """,
            "Trend_Indicators": """
                CREATE TABLE IF NOT EXISTS Trend_Indicators (
                    ticker TEXT,
                    parabolic_sar REAL, -- Parabolic SAR
                    directional_movement_index REAL, -- Directional Movement Index
                    minus_directional_indicator REAL, -- Minus Directional Indicator
                    plus_directional_indicator REAL, -- Plus Directional Indicator
                    average_directional_index REAL, -- Average Directional Index
                    adx REAL,  -- ADX
                    adx_pos_di REAL,  -- ADX Positive DI
                    adx_neg_di REAL,  -- ADX Negative DI
                    cci REAL,  -- CCI
                    macd REAL,  -- MACD
                    macd_signal REAL,  -- MACD Signal Line
                    macd_diff REAL,  -- MACD Difference
                    ema_fast REAL,  -- EMA Fast
                    ema_slow REAL,  -- EMA Slow
                    ichimoku_a REAL,  -- Ichimoku A
                    ichimoku_b REAL,  -- Ichimoku B
                    ichimoku_base_line REAL,  -- Ichimoku Base Line
                    ichimoku_conversion_line REAL,  -- Ichimoku Conversion Line
                    kst REAL,  -- KST
                    kst_sig REAL,  -- KST Signal Line
                    kst_diff REAL,  -- KST Difference
                    psar REAL,  -- PSAR
                    psar_up_indicator REAL,  -- PSAR Up Indicator
                    psar_down_indicator REAL,  -- PSAR Down Indicator
                    stc REAL,  -- STC
                    trix REAL,  -- Trix
                    vortex_ind_pos REAL,  -- Vortex Indicator Positive DI
                    vortex_ind_neg REAL,  -- Vortex Indicator Negative DI
                    vortex_ind_diff REAL  -- Vortex Indicator Difference
                )
            """,
            "Price_Patterns_Candlesticks": """
                CREATE TABLE IF NOT EXISTS Price_Patterns_Candlesticks (
                    ticker TEXT,
                    cdl_2_crows REAL,  -- CDL 2 Crows
                    cdl_3_black_crows REAL,  -- CDL 3 Black Crows
                    cdl_3_inside REAL,  -- CDL 3 Inside
                    cdl_3_line_strike REAL,  -- CDL 3 Line Strike
                    cdl_3_outside REAL,  -- CDL 3 Outside
                    cdl_3_stars_in_south REAL,  -- CDL 3 Stars In South
                    cdl_3_white_soldiers REAL,  -- CDL 3 White Soldiers
                    cdl_abandoned_baby REAL,  -- CDL Abandoned Baby
                    cdl_advance_block REAL,  -- CDL Advance Block
                    cdl_belt_hold REAL,  -- CDL Belt Hold
                    cdl_breakaway REAL,  -- CDL Breakaway
                    cdl_closing_marubozu REAL,  -- CDL Closing Marubozu
                    cdl_conceal_baby_swall REAL,  -- CDL Conceal Baby Swall
                    cdl_counterattack REAL,  -- CDL Counterattack
                    cdl_dark_cloud_cover REAL,  -- CDL Dark Cloud Cover
                    cdl_doji REAL,  -- CDL Doji
                    cdl_doji_star REAL,  -- CDL Doji Star
                    cdl_dragonfly_doji REAL,  -- CDL Dragonfly Doji
                    cdl_engulfing REAL,  -- CDL Engulfing
                    cdl_evening_doji_star REAL,  -- CDL Evening Doji Star
                    cdl_evening_star REAL,  -- CDL Evening Star
                    cdl_gap_side_side_white REAL,  -- CDL Gap Side Side White
                    cdl_gravestone_doji REAL,  -- CDL Gravestone Doji
                    cdl_hammer REAL,  -- CDL Hammer
                    cdl_hanging_man REAL,  -- CDL Hanging Man
                    cdl_harami REAL,  -- CDL Harami
                    cdl_harami_cross REAL,  -- CDL Harami Cross
                    cdl_high_wave REAL,  -- CDL High Wave
                    cdl_hikkake REAL,  -- CDL Hikkake
                    cdl_hikkake_modified REAL,  -- CDL Hikkake Modified
                    cdl_homing_pigeon REAL,  -- CDL Homing Pigeon
                    cdl_identical_3_crows REAL,  -- CDL Identical 3 Crows
                    cdl_in_neck REAL,  -- CDL In Neck
                    cdl_inverted_hammer REAL,  -- CDL Inverted Hammer
                    cdl_kicking REAL,  -- CDL Kicking
                    cdl_kicking_by_length REAL,  -- CDL Kicking By Length
                    cdl_ladder_bottom REAL,  -- CDL Ladder Bottom
                    cdl_long_legged_doji REAL,  -- CDL Long Legged Doji
                    cdl_long_line REAL,  -- CDL Long Line
                    cdl_marubozu REAL,  -- CDL Marubozu
                    cdl_matching_low REAL,  -- CDL Matching Low
                    cdl_mat_hold REAL,  -- CDL Mat Hold
                    cdl_morning_doji_star REAL,  -- CDL Morning Doji Star
                    cdl_morning_star REAL,  -- CDL Morning Star
                    cdl_on_neck REAL,  -- CDL On Neck
                    cdl_piercing REAL,  -- CDL Piercing
                    cdl_rickshaw_man REAL,  -- CDL Rickshaw Man
                    cdl_rise_fall_3_methods REAL,  -- CDL Rise Fall 3 Methods
                    cdl_separating_lines REAL,  -- CDL Separating Lines
                    cdl_shooting_star REAL,  -- CDL Shooting Star
                    cdl_short_line REAL,  -- CDL Short Line
                    cdl_spinning_top REAL,  -- CDL Spinning Top
                    cdl_stalled_pattern REAL,  -- CDL Stalled Pattern
                    cdl_stick_sandwich REAL,  -- CDL Stick Sandwich
                    cdl_takuri REAL,  -- CDL Takuri
                    cdl_tasuki_gap REAL,  -- CDL Tasuki Gap
                    cdl_thrusting REAL,  -- CDL Thrusting
                    cdl_tristar REAL,  -- CDL Tristar
                    cdl_unique_3_river REAL,  -- CDL Unique 3 River
                    cdl_upside_gap_2_crows REAL,  -- CDL Upside Gap 2 Crows
                    cdl_x_side_gap_3_methods REAL  -- CDL X Side Gap 3 Methods
                )
            """,
            "Advanced_Statistical_Measures": """
                CREATE TABLE IF NOT EXISTS Advanced_Statistical_Measures (
                    ticker TEXT,
                    beta REAL,  -- Beta
                    correlation_coefficient REAL,  -- Correlation Coefficient
                    linear_regression_angle REAL,  -- Linear Regression Angle
                    linear_regression_intercept REAL,  -- Linear Regression Intercept
                    linear_regression_slope REAL,  -- Linear Regression Slope
                    standard_deviation REAL,  -- Standard Deviation
                    standard_error REAL,  -- Standard Error
                    time_series_forecast REAL,  -- Time Series Forecast
                    variance REAL  -- Variance
                )
            """,
            "Math_Transformations": """
                CREATE TABLE IF NOT EXISTS Math_Transformations (
                    ticker TEXT,
                    transform_acos REAL,  -- Transform ACOS
                    transform_asin REAL,  -- Transform ASIN
                    transform_atan REAL,  -- Transform ATAN
                    transform_ceil REAL,  -- Transform CEIL
                    transform_cos REAL,  -- Transform COS
                    transform_cosh REAL,  -- Transform COSH
                    transform_exp REAL,  -- Transform EXP
                    transform_floor REAL,  -- Transform FLOOR
                    transform_ln REAL,  -- Transform LN
                    transform_log10 REAL,  -- Transform LOG10
                    transform_sin REAL,  -- Transform SIN
                    transform_sinh REAL,  -- Transform SINH
                    transform_sqrt REAL,  -- Transform SQRT
                    transform_tan REAL,  -- Transform TAN
                )
            """,
            # Add similar queries for all other tables.
            # Due to the vast number of tables, only a few are showcased here.
            # Each table creation string follows the pattern established.
        }
        for table, query in tables.items():
            self.execute_query(query)

    def update_table(self, table: str, data: Dict[str, Any]):
        """Update a specific table with a dictionary of data."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.execute_query(query, tuple(data.values()))

    def fetch_data(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch data from the database."""
        self.cursor.execute(query, params)
        columns = [column[0] for column in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]


# Example usage
db = StockDatabase(DB_PATH)


"""




Table: Moving_Averages
        "ticker", # Stock ticker
        "moving_average_5",  # Moving Average 5
        "moving_average_15",  # Moving Average 15
        "moving_average_50",  # Moving Average 50
        "moving_average_200",  # Moving Average 200
        "moving_average_365",  # Moving Average 365
        "exponential_moving_average_5",  # Exponential Moving Average 5
        "exponential_moving_average_15",  # Exponential Moving Average 15
        "exponential_moving_average_50",  # Exponential Moving Average 50
        "exponential_moving_average_200",  # Exponential Moving Average 200
        "exponential_moving_average_365",  # Exponential Moving Average 365
        "multiface_moving_average_5",  # Multiface Moving Average 5
        "multiface_moving_average_15",  # Multiface Moving Average 15
        "multiface_moving_average_50",  # Multiface Moving Average 50
        "multiface_moving_average_200",  # Multiface Moving Average 200
        "multiface_moving_average_365",  # Multiface Moving Average 365
        "triple_exponential_moving_average_5",  # Triple Exponential Moving Average 5
        "triple_exponential_moving_average_15",  # Triple Exponential Moving Average 15
        "triple_exponential_moving_average_50",  # Triple Exponential Moving Average 50
        "triple_exponential_moving_average_200",  # Triple Exponential Moving Average 200
        "triple_exponential_moving_average_365",  # Triple Exponential Moving Average 365

Table: Oscillators_Momentum
        "ticker"
        "momentum_rsi_14",  # Momentum RSI
        "momentum_mfi_14",  # Momentum MFI
        "momentum_tsi_25_13",  # Momentum TSI
        "momentum_uo_7_14_28",  # Momentum UO
        "momentum_stoch_rsi_14",  # Momentum Stoch RSI
        "momentum_stoch_rsi_k_14",  # Momentum Stoch RSI %K
        "momentum_stoch_rsi_d_14",  # Momentum Stoch RSI %D
        "momentum_wr_14",  # Momentum Williams %R
        "momentum_ao",  # Momentum Awesome Oscillator
        "momentum_kama_10_2_30",  # Momentum KAMA
        "momentum_ppo_12_26_9",  # Momentum PPO
        "momentum_ppo_signal_12_26_9",  # Momentum PPO Signal Line
        "momentum_ppo_hist_12_26_9",  # Momentum PPO Histogram
        "momentum_pvo_12_26_9",  # Momentum PVO
        "momentum_pvo_signal_12_26_9",  # Momentum PVO Signal Line
        "momentum_pvo_hist_12_26_9",  # Momentum PVO Histogram
        "momentum_roc_10",  # Momentum ROC
        "momentum_roc_100",  # Momentum ROC 100
        "momentum_roc_100_sma_10",  # Momentum ROC 100 SMA 10

Table: Volatility_Indicators
        "ticker", # Stock ticker
        "volatility_bbands_20_2",  # Volatility Bollinger Bands
        "volatility_bbands_upper_20_2",  # Volatility Bollinger Bands Upper Band
        "volatility_bbands_lower_20_2",  # Volatility Bollinger Bands Lower Band
        "volatility_bbands_middle_20_2",  # Volatility Bollinger Bands Middle Band
        "volatility_kc_20_2",  # Volatility Keltner Channels
        "volatility_kc_upper_20_2",  # Volatility Keltner Channels Upper Band
        "volatility_kc_lower_20_2",  # Volatility Keltner Channels Lower Band
        "volatility_kc_middle_20_2",  # Volatility Keltner Channels Middle Band
        "volatility_dch_20",  # Volatility Donchian Channels
        "volatility_dch_upper_20",  # Volatility Donchian Channels Upper Band
        "volatility_dch_lower_20",  # Volatility Donchian Channels Lower Band
        "volatility_dch_middle_20",  # Volatility Donchian Channels Middle Band
        "volatility_atr_14",  # Volatility ATR
        "volatility_true_range_14",  # Volatility True Range
        "volatility_natr_14",  # Volatility NATR

Table: Volume_Indicators
        "ticker", # Stock ticker
        "volume_adi",  # Volume ADI
        "volume_obv",  # Volume OBV
        "volume_cmf_20",  # Volume CMF
        "volume_fi_13",  # Volume Force Index
        "volume_em_14_100000000",  # Volume Ease of Movement
        "volume_sma_em_14_100000000",  # Volume SMA Ease of Movement
        "volume_vpt",  # Volume VPT
        "volume_nvi",  # Volume NVI
        "volume_vwap",  # Volume VWAP

Table: Trend_Indicators
        "ticker", # Stock ticker
        "parabolic_sar_0_02_0_2", # Parabolic SAR
        "directional_movement_index_14", # Directional Movement Index
        "minus_directional_indicator_14", # Minus Directional Indicator
        "plus_directional_indicator_14", # Plus Directional Indicator
        "average_directional_index_14", # Average Directional Index
        "trend_adx_14",  # Trend ADX 14
        "trend_adx_pos_di_14",  # Trend ADX Positive DI
        "trend_adx_neg_di_14",  # Trend ADX Negative DI
        "trend_cci_14",  # Trend CCI
        "trend_macd_12_26_9",  # Trend MACD
        "trend_macd_signal_12_26_9",  # Trend MACD Signal Line
        "trend_macd_diff_12_26_9",  # Trend MACD Difference
        "trend_ema_fast_12",  # Trend EMA Fast
        "trend_ema_slow_26",  # Trend EMA Slow
        "trend_ichimoku_a_9_26_52",  # Trend Ichimoku A
        "trend_ichimoku_b_9_26_52",  # Trend Ichimoku B
        "trend_ichimoku_base_line_9_26_52",  # Trend Ichimoku Base Line
        "trend_ichimoku_conversion_line_9_26_52",  # Trend Ichimoku Conversion Line
        "trend_kst_10_15_20_30",  # Trend KST
        "trend_kst_sig_10_15_20_30",  # Trend KST Signal Line
        "trend_kst_diff_10_15_20_30",  # Trend KST Difference
        "trend_psar_0_02_0_2",  # Trend PSAR
        "trend_psar_up_indicator",  # Trend PSAR Up Indicator
        "trend_psar_down_indicator",  # Trend PSAR Down Indicator
        "trend_stc_10_12_26_9",  # Trend STC
        "trend_trix_30_9",  # Trend Trix
        "trend_vortex_ind_pos_14",  # Trend Vortex Indicator Positive DI
        "trend_vortex_ind_neg_14",  # Trend Vortex Indicator Negative DI
        "trend_vortex_ind_diff_14",  # Trend Vortex Indicator Difference

Table: Price_Patterns_Candlesticks
        "ticker", # Stock
        "pattern_recognition_cdl2crows",  # Pattern Recognition CDL 2 Crows
        "pattern_recognition_cdl3blackcrows",  # Pattern Recognition CDL 3 Black Crows
        "pattern_recognition_cdl3inside",  # Pattern Recognition CDL 3 Inside
        "pattern_recognition_cdl3linestrike",  # Pattern Recognition CDL 3 Line Strike
        "pattern_recognition_cdl3outside",  # Pattern Recognition CDL 3 Outside
        "pattern_recognition_cdl3starsinsouth",  # Pattern Recognition CDL 3 Stars In South
        "pattern_recognition_cdl3whitesoldiers",  # Pattern Recognition CDL 3 White Soldiers
        "pattern_recognition_cdlabandonedbaby",  # Pattern Recognition CDL Abandoned Baby
        "pattern_recognition_cdladvanceblock",  # Pattern Recognition CDL Advance Block
        "pattern_recognition_cdlbelthold",  # Pattern Recognition CDL Belt Hold
        "pattern_recognition_cdlbreakaway",  # Pattern Recognition CDL Breakaway
        "pattern_recognition_cdlclosingmarubozu",  # Pattern Recognition CDL Closing Marubozu
        "pattern_recognition_cdlconcealbabyswall",  # Pattern Recognition CDL Conceal Baby Swall
        "pattern_recognition_cdlcounterattack",  # Pattern Recognition CDL Counterattack
        "pattern_recognition_cdldarkcloudcover",  # Pattern Recognition CDL Dark Cloud Cover
        "pattern_recognition_cdldoji",  # Pattern Recognition CDL Doji
        "pattern_recognition_cdldojistar",  # Pattern Recognition CDL Doji Star
        "pattern_recognition_cdldragonflydoji",  # Pattern Recognition CDL Dragonfly Doji
        "pattern_recognition_cdlengulfing",  # Pattern Recognition CDL Engulfing
        "pattern_recognition_cdleveningdojistar",  # Pattern Recognition CDL Evening Doji Star
        "pattern_recognition_cdleveningstar",  # Pattern Recognition CDL Evening Star
        "pattern_recognition_cdlgapsidesidewhite",  # Pattern Recognition CDL Gap Side Side White
        "pattern_recognition_cdlgravestonedoji",  # Pattern Recognition CDL Gravestone Doji
        "pattern_recognition_cdlhammer",  # Pattern Recognition CDL Hammer
        "pattern_recognition_cdlhangingman",  # Pattern Recognition CDL Hanging Man
        "pattern_recognition_cdlharami",  # Pattern Recognition CDL Harami
        "pattern_recognition_cdlharamicross",  # Pattern Recognition CDL Harami Cross
        "pattern_recognition_cdlhighwave",  # Pattern Recognition CDL High Wave
        "pattern_recognition_cdlhikkake",  # Pattern Recognition CDL Hikkake
        "pattern_recognition_cdlhikkakemod",  # Pattern Recognition CDL Hikkake Modified
        "pattern_recognition_cdlhomingpigeon",  # Pattern Recognition CDL Homing Pigeon
        "pattern_recognition_cdlidentical3crows",  # Pattern Recognition CDL Identical 3 Crows
        "pattern_recognition_cdlinneck",  # Pattern Recognition CDL In Neck
        "pattern_recognition_cdlinvertedhammer",  # Pattern Recognition CDL Inverted Hammer
        "pattern_recognition_cdlkicking",  # Pattern Recognition CDL Kicking
        "pattern_recognition_cdlkickingbylength",  # Pattern Recognition CDL Kicking By Length
        "pattern_recognition_cdlladderbottom",  # Pattern Recognition CDL Ladder Bottom
        "pattern_recognition_cdllongleggeddoji",  # Pattern Recognition CDL Long Legged Doji
        "pattern_recognition_cdllongline",  # Pattern Recognition CDL Long Line
        "pattern_recognition_cdlmarubozu",  # Pattern Recognition CDL Marubozu
        "pattern_recognition_cdlmatchinglow",  # Pattern Recognition CDL Matching Low
        "pattern_recognition_cdlmathold",  # Pattern Recognition CDL Mat Hold
        "pattern_recognition_cdlmorningdojistar",  # Pattern Recognition CDL Morning Doji Star
        "pattern_recognition_cdlmorningstar",  # Pattern Recognition CDL Morning Star
        "pattern_recognition_cdlonneck",  # Pattern Recognition CDL On Neck
        "pattern_recognition_cdlpiercing",  # Pattern Recognition CDL Piercing
        "pattern_recognition_cdlrickshawman",  # Pattern Recognition CDL Rickshaw Man
        "pattern_recognition_cdlrisefall3methods",  # Pattern Recognition CDL Rise Fall 3 Methods
        "pattern_recognition_cdlseparatinglines",  # Pattern Recognition CDL Separating Lines
        "pattern_recognition_cdlshootingstar",  # Pattern Recognition CDL Shooting Star
        "pattern_recognition_cdlshortline",  # Pattern Recognition CDL Short Line
        "pattern_recognition_cdlspinningtop",  # Pattern Recognition CDL Spinning Top
        "pattern_recognition_cdlstalledpattern",  # Pattern Recognition CDL Stalled Pattern
        "pattern_recognition_cdlsticksandwich",  # Pattern Recognition CDL Stick Sandwich
        "pattern_recognition_cdltakuri",  # Pattern Recognition CDL Takuri
        "pattern_recognition_cdltasukigap",  # Pattern Recognition CDL Tasuki Gap
        "pattern_recognition_cdlthrusting",  # Pattern Recognition CDL Thrusting
        "pattern_recognition_cdltristar",  # Pattern Recognition CDL Tristar
        "pattern_recognition_cdlunique3river",  # Pattern Recognition CDL Unique 3 River
        "pattern_recognition_cdlupsidegap2crows",  # Pattern Recognition CDL Upside Gap 2 Crows
        "pattern_recognition_cdlxsidegap3methods",  # Pattern Recognition CDL X Side Gap 3 Methods

Table: Advanced_Statistical_Measures
        "ticker", # Stock
        "statistic_beta",  # Statistic Beta
        "statistic_correlation_coefficient",  # Statistic Correlation Coefficient
        "statistic_linear_regression_angle",  # Statistic Linear Regression Angle
        "statistic_linear_regression_intercept",  # Statistic Linear Regression Intercept
        "statistic_linear_regression_slope",  # Statistic Linear Regression Slope
        "statistic_standard_deviation",  # Statistic Standard Deviation
        "statistic_standard_error",  # Statistic Standard Error
        "statistic_time_series_forecast",  # Statistic Time Series Forecast
        "statistic_variance",  # Statistic Variance

Table: Mathematical_Transformations
        "ticker" # Stock
        "math_transform_acos",  # Math Transform ACOS
        "math_transform_asin",  # Math Transform ASIN
        "math_transform_atan",  # Math Transform ATAN
        "math_transform_ceil",  # Math Transform CEIL
        "math_transform_cos",  # Math Transform COS
        "math_transform_cosh",  # Math Transform COSH
        "math_transform_exp",  # Math Transform EXP
        "math_transform_floor",  # Math Transform FLOOR
        "math_transform_ln",  # Math Transform LN
        "math_transform_log10",  # Math Transform LOG10
        "math_transform_sin",  # Math Transform SIN
        "math_transform_sinh",  # Math Transform SINH
        "math_transform_sqrt",  # Math Transform SQRT
        "math_transform_tan",  # Math Transform TAN
        "math_transform_tanh",  # Math Transform TANH
        "math_transform_add",  # Math Transform ADD
        "math_transform_div",  # Math Transform DIV
        "math_transform_max",  # Math Transform MAX
        "math_transform_maxindex",  # Math Transform MAXINDEX
        "math_transform_min",  # Math Transform MIN
        "math_transform_minindex",  # Math Transform MININDEX
        "math_transform_minmax",  # Math Transform MINMAX
        "math_transform_minmaxindex",  # Math Transform MINMAXINDEX
        "math_transform_mult",  # Math Transform MULT
        "math_transform_sub",  # Math Transform SUB
        "math_transform_sum",  # Math Transform SUM
        "math_operator_abs",  # Math Operator ABS
        "math_operator_acos",  # Math Operator ACOS
        "math_operator_add",  # Math Operator ADD
        "math_operator_asin",  # Math Operator ASIN
        "math_operator_atan",  # Math Operator ATAN
        "math_operator_ceil",  # Math Operator CEIL
        "math_operator_cos",  # Math Operator COS
        "math_operator_cosh",  # Math Operator COSH
        "math_operator_div",  # Math Operator DIV
        "math_operator_exp",  # Math Operator EXP
        "math_operator_floor",  # Math Operator FLOOR
        "math_operator_ln",  # Math Operator LN
        "math_operator_log10",  # Math Operator LOG10
        "math_operator_max",  # Math Operator MAX
        "math_operator_maxindex",  # Math Operator MAXINDEX
        "math_operator_min",  # Math Operator MIN
        "math_operator_minindex",  # Math Operator MININDEX
        "math_operator_minmax",  # Math Operator MINMAX
        "math_operator_minmaxindex",  # Math Operator MINMAXINDEX
        "math_operator_mult",  # Math Operator MULT
        "math_operator_round",  # Math Operator ROUND
        "math_operator_sin",  # Math Operator SIN
        "math_operator_sinh",  # Math Operator SINH
        "math_operator_sqrt",  # Math Operator SQRT
        "math_operator_sub",  # Math Operator SUB
        "math_operator_sum",  # Math Operator SUM
        "math_operator_tan",  # Math Operator TAN
        "math_operator_tanh",  # Math Operator TANH
        "math_operator_todeg",  # Math Operator TO DEG
        "math_operator_torad",  # Math Operator TO RAD
        "math_operator_trunc",  # Math Operator TRUNC

Table: Corporate_Events
        "ticker", # Stock ticker
        "event_type", # e.g., earnings release, product launch
        "event_date", # Date of the event
        "impact_score", # Quantitative measure of expected impact

Table: Market_Sentiment
        "ticker", # Stock ticker
        "sentiment_score", # Sentiment score
        "sentiment_volume", # Sentiment volume
        "date" # Date of the sentiment

Table: Risk_Metrics
        "ticker", # Stock ticker
        "beta", # Beta
        "alpha", # Alpha
        "sharpe_ratio", # Sharpe ratio
        "sortino_ratio", # Sortino ratio
        "date" # Date of the risk metrics

Table: Trading_Sessions
        "ticker", # Stock ticker
        "session_date", # Date of the trading session
        "open_price", # Opening price
        "close_price", # Closing price
        "session_high", # Highest price during the session
        "session_low", # Lowest price during the session
        "transaction_volume" # Volume of transactions during the session

Table: Derivatives_Options
        "ticker", # Stock ticker
        "option_type", # Call or put
        "strike_price", # Price at which the option can be exercised
        "expiration_date", # Date when the option expires
        "open_interest", # Number of open contracts
        "implied_volatility" # Measure of expected volatility

Table: Sector_Industry
        "ticker", # Stock ticker
        "sector", # Sector of the stock
        "industry", # Industry of the stock
        "market_cap", # Market capitalization
        "employee_count" # Number of employees

Table: Financial_Ratios
        "ticker", # Stock ticker
        "price_to_earnings", # Price-to-earnings ratio
        "return_on_equity", # Return on equity
        "debt_to_equity", # Debt-to-equity ratio
        "current_ratio", # Current ratio
        "date" # Date of the financial ratios


"""
