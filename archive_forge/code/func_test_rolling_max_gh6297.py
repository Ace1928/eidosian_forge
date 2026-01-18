from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_rolling_max_gh6297(step):
    """Replicate result expected in GH #6297"""
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    indices.append(datetime(1975, 1, 3, 6, 0))
    series = Series(range(1, 7), index=indices)
    series = series.map(lambda x: float(x))
    series = series.sort_index()
    expected = Series([1.0, 2.0, 6.0, 4.0, 5.0], index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq='D'))[::step]
    x = series.resample('D').max().rolling(window=1, step=step).max()
    tm.assert_series_equal(expected, x)