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
class IndentLoggerAdapter(logging.LoggerAdapter):

    def process(self, msg, kwargs):
        if get_yf_logger().isEnabledFor(logging.DEBUG):
            i = ' ' * self.extra['indent']
            if not isinstance(msg, str):
                msg = str(msg)
            msg = '\n'.join([i + m for m in msg.split('\n')])
        return (msg, kwargs)