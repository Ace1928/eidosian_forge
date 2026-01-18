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
def test_resample_rounding(unit):
    ts = ['2014-11-08 00:00:01', '2014-11-08 00:00:02', '2014-11-08 00:00:02', '2014-11-08 00:00:03', '2014-11-08 00:00:07', '2014-11-08 00:00:07', '2014-11-08 00:00:08', '2014-11-08 00:00:08', '2014-11-08 00:00:08', '2014-11-08 00:00:09', '2014-11-08 00:00:10', '2014-11-08 00:00:11', '2014-11-08 00:00:11', '2014-11-08 00:00:13', '2014-11-08 00:00:14', '2014-11-08 00:00:15', '2014-11-08 00:00:17', '2014-11-08 00:00:20', '2014-11-08 00:00:21']
    df = DataFrame({'value': [1] * 19}, index=pd.to_datetime(ts))
    df.index = df.index.as_unit(unit)
    result = df.resample('6s').sum()
    expected = DataFrame({'value': [4, 9, 4, 2]}, index=date_range('2014-11-08', freq='6s', periods=4).as_unit(unit))
    tm.assert_frame_equal(result, expected)
    result = df.resample('7s').sum()
    expected = DataFrame({'value': [4, 10, 4, 1]}, index=date_range('2014-11-08', freq='7s', periods=4).as_unit(unit))
    tm.assert_frame_equal(result, expected)
    result = df.resample('11s').sum()
    expected = DataFrame({'value': [11, 8]}, index=date_range('2014-11-08', freq='11s', periods=2).as_unit(unit))
    tm.assert_frame_equal(result, expected)
    result = df.resample('13s').sum()
    expected = DataFrame({'value': [13, 6]}, index=date_range('2014-11-08', freq='13s', periods=2).as_unit(unit))
    tm.assert_frame_equal(result, expected)
    result = df.resample('17s').sum()
    expected = DataFrame({'value': [16, 3]}, index=date_range('2014-11-08', freq='17s', periods=2).as_unit(unit))
    tm.assert_frame_equal(result, expected)