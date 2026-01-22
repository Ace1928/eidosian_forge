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
class IndentationContext:

    def __init__(self, increment=1):
        self.increment = increment

    def __enter__(self):
        _indentation_level.indent = getattr(_indentation_level, 'indent', 0) + self.increment

    def __exit__(self, exc_type, exc_val, exc_tb):
        _indentation_level.indent -= self.increment