import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
def test_rolling_forward_skewness(frame_or_series, step):
    values = np.arange(10.0)
    values[5] = 100.0
    indexer = FixedForwardWindowIndexer(window_size=5)
    rolling = frame_or_series(values).rolling(window=indexer, min_periods=3, step=step)
    result = rolling.skew()
    expected = frame_or_series([0.0, 2.232396, 2.229508, 2.22834, 2.229091, 2.231989, 0.0, 0.0, np.nan, np.nan])[::step]
    tm.assert_equal(result, expected)