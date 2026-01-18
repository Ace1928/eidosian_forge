import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import ccxt  # For connecting to various trading exchanges
import backtrader as bt  # For backtesting
import asyncio
import aiohttp
import websocket
import logging
import yfinance as yf  # For downloading market data from Yahoo Finance
def test_strategies(self, symbols: list, strategy: str):
    """Test a strategy across multiple symbols."""
    results = {}
    for symbol in symbols:
        self.fetch_data(symbol)
        self.apply_strategy(strategy)
        results[symbol] = self.evaluate_performance()
    return results