from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_resample_ohlc_dataframe(unit):
    df = DataFrame({'PRICE': {Timestamp('2011-01-06 10:59:05', tz=None): 24990, Timestamp('2011-01-06 12:43:33', tz=None): 25499, Timestamp('2011-01-06 12:54:09', tz=None): 25499}, 'VOLUME': {Timestamp('2011-01-06 10:59:05', tz=None): 1500000000, Timestamp('2011-01-06 12:43:33', tz=None): 5000000000, Timestamp('2011-01-06 12:54:09', tz=None): 100000000}}).reindex(['VOLUME', 'PRICE'], axis=1)
    df.index = df.index.as_unit(unit)
    df.columns.name = 'Cols'
    res = df.resample('h').ohlc()
    exp = pd.concat([df['VOLUME'].resample('h').ohlc(), df['PRICE'].resample('h').ohlc()], axis=1, keys=df.columns)
    assert exp.columns.names[0] == 'Cols'
    tm.assert_frame_equal(exp, res)
    df.columns = [['a', 'b'], ['c', 'd']]
    res = df.resample('h').ohlc()
    exp.columns = pd.MultiIndex.from_tuples([('a', 'c', 'open'), ('a', 'c', 'high'), ('a', 'c', 'low'), ('a', 'c', 'close'), ('b', 'd', 'open'), ('b', 'd', 'high'), ('b', 'd', 'low'), ('b', 'd', 'close')])
    tm.assert_frame_equal(exp, res)