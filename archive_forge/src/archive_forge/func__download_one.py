from __future__ import print_function
import logging
import time as _time
import traceback
import multitasking as _multitasking
import pandas as _pd
from . import Ticker, utils
from .data import YfData
from . import shared
def _download_one(ticker, start=None, end=None, auto_adjust=False, back_adjust=False, repair=False, actions=False, period='max', interval='1d', prepost=False, proxy=None, rounding=False, keepna=False, timeout=10):
    data = None
    try:
        data = Ticker(ticker).history(period=period, interval=interval, start=start, end=end, prepost=prepost, actions=actions, auto_adjust=auto_adjust, back_adjust=back_adjust, repair=repair, proxy=proxy, rounding=rounding, keepna=keepna, timeout=timeout, raise_errors=True)
    except Exception as e:
        shared._DFS[ticker.upper()] = utils.empty_df()
        shared._ERRORS[ticker.upper()] = repr(e)
        shared._TRACEBACKS[ticker.upper()] = traceback.format_exc()
    else:
        shared._DFS[ticker.upper()] = data
    return data