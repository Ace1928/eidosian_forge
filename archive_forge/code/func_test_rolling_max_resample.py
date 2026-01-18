from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_rolling_max_resample(step):
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    indices.append(datetime(1975, 1, 5, 1))
    indices.append(datetime(1975, 1, 5, 2))
    series = Series(list(range(5)) + [10, 20], index=indices)
    series = series.map(lambda x: float(x))
    series = series.sort_index()
    expected = Series([0.0, 1.0, 2.0, 3.0, 20.0], index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq='D'))[::step]
    x = series.resample('D').max().rolling(window=1, step=step).max()
    tm.assert_series_equal(expected, x)
    expected = Series([0.0, 1.0, 2.0, 3.0, 10.0], index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq='D'))[::step]
    x = series.resample('D').median().rolling(window=1, step=step).max()
    tm.assert_series_equal(expected, x)
    v = (4.0 + 10.0 + 20.0) / 3.0
    expected = Series([0.0, 1.0, 2.0, 3.0, v], index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq='D'))[::step]
    x = series.resample('D').mean().rolling(window=1, step=step).max()
    tm.assert_series_equal(expected, x)