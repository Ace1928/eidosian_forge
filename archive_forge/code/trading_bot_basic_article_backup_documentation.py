import pandas as pd  # Data manipulation
import requests  # HTTP requests
import pandas_ta as ta  # Technical analysis
import matplotlib as mpl  # Plotting
import matplotlib.pyplot as plt  # Plotting
from termcolor import colored as cl  # Text customization
import math  # Mathematical operations
import numpy as np  # Numerical operations
from datetime import datetime as dt  # Date and time operations
from typing import (
import sqlite3  # Database operations
import yfinance as yf  # Yahoo Finance API
from sqlite3 import Connection, Cursor
from typing import Optional  # Type hinting
import seaborn as sns  # Data visualization
import logging  # Logging
import time  # Time operations
import sys  # System-specific parameters and functions
from scripts.trading_bot.indecache import async_cache  # Async cache decorator

        Returns the sleep time in seconds based on the given frequency.

        Args:
            frequency (str): The frequency of data updates.

        Returns:
            int: The sleep time in seconds.
        