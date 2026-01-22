import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
from yfinance import shared, utils
from yfinance.const import _BASE_URL_, _PRICE_COLNAMES_
@utils.log_indent_decorator
def _fix_unit_mixups(self, df, interval, tz_exchange, prepost):
    if df.empty:
        return df
    df2 = self._fix_unit_switch(df, interval, tz_exchange)
    df3 = self._fix_unit_random_mixups(df2, interval, tz_exchange, prepost)
    return df3