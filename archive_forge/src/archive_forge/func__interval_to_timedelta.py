from __future__ import print_function
import datetime as _datetime
import logging
import re as _re
import sys as _sys
import threading
from functools import lru_cache
from inspect import getmembers
from types import FunctionType
from typing import List, Optional
import numpy as _np
import pandas as _pd
import pytz as _tz
import requests as _requests
from dateutil.relativedelta import relativedelta
from pytz import UnknownTimeZoneError
from yfinance import const
from .const import _BASE_URL_
def _interval_to_timedelta(interval):
    if interval == '1mo':
        return relativedelta(months=1)
    elif interval == '3mo':
        return relativedelta(months=3)
    elif interval == '1y':
        return relativedelta(years=1)
    elif interval == '1wk':
        return _pd.Timedelta(days=7)
    else:
        return _pd.Timedelta(interval)