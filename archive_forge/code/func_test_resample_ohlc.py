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
def test_resample_ohlc(series, unit):
    s = series
    s.index = s.index.as_unit(unit)
    grouper = Grouper(freq=Minute(5))
    expect = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    result = s.resample('5Min').ohlc()
    assert len(result) == len(expect)
    assert len(result.columns) == 4
    xs = result.iloc[-2]
    assert xs['open'] == s.iloc[-6]
    assert xs['high'] == s[-6:-1].max()
    assert xs['low'] == s[-6:-1].min()
    assert xs['close'] == s.iloc[-2]
    xs = result.iloc[0]
    assert xs['open'] == s.iloc[0]
    assert xs['high'] == s[:5].max()
    assert xs['low'] == s[:5].min()
    assert xs['close'] == s.iloc[4]