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
def _reindex_events(df, new_index, data_col_name):
    if len(new_index) == len(set(new_index)):
        df.index = new_index
        return df
    df['_NewIndex'] = new_index
    if data_col_name in ['Dividends', 'Capital Gains']:
        df = df.groupby('_NewIndex').sum()
        df.index.name = None
    elif data_col_name == 'Stock Splits':
        df = df.groupby('_NewIndex').prod()
        df.index.name = None
    else:
        raise Exception(f"New index contains duplicates but unsure how to aggregate for '{data_col_name}'")
    if '_NewIndex' in df.columns:
        df = df.drop('_NewIndex', axis=1)
    return df