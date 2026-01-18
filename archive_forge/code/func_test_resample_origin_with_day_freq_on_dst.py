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
def test_resample_origin_with_day_freq_on_dst(unit):
    tz = 'America/Chicago'

    def _create_series(values, timestamps, freq='D'):
        return Series(values, index=DatetimeIndex([Timestamp(t, tz=tz) for t in timestamps], freq=freq, ambiguous=True).as_unit(unit))
    start = Timestamp('2013-11-02', tz=tz)
    end = Timestamp('2013-11-03 23:59', tz=tz)
    rng = date_range(start, end, freq='1h').as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)
    expected = _create_series([24.0, 25.0], ['2013-11-02', '2013-11-03'])
    for origin in ['epoch', 'start', 'start_day', start, None]:
        result = ts.resample('D', origin=origin).sum()
        tm.assert_series_equal(result, expected)
    start = Timestamp('2013-11-03', tz=tz)
    end = Timestamp('2013-11-03 23:59', tz=tz)
    rng = date_range(start, end, freq='1h').as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)
    expected_ts = ['2013-11-02 22:00-05:00', '2013-11-03 22:00-06:00']
    expected = _create_series([23.0, 2.0], expected_ts)
    result = ts.resample('D', origin='start', offset='-2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 22:00-05:00', '2013-11-03 21:00-06:00']
    expected = _create_series([22.0, 3.0], expected_ts, freq='24h')
    result = ts.resample('24h', origin='start', offset='-2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 02:00-05:00', '2013-11-03 02:00-06:00']
    expected = _create_series([3.0, 22.0], expected_ts)
    result = ts.resample('D', origin='start', offset='2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 23:00-05:00', '2013-11-03 23:00-06:00']
    expected = _create_series([24.0, 1.0], expected_ts)
    result = ts.resample('D', origin='start', offset='-1h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 01:00-05:00', '2013-11-03 01:00:00-0500']
    expected = _create_series([1.0, 24.0], expected_ts)
    result = ts.resample('D', origin='start', offset='1h').sum()
    tm.assert_series_equal(result, expected)