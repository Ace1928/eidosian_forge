import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_ragged_count(self, ragged):
    df = ragged
    result = df.rolling(window='1s', min_periods=1).count()
    expected = df.copy()
    expected['B'] = [1.0, 1, 1, 1, 1]
    tm.assert_frame_equal(result, expected)
    df = ragged
    result = df.rolling(window='1s').count()
    tm.assert_frame_equal(result, expected)
    result = df.rolling(window='2s', min_periods=1).count()
    expected = df.copy()
    expected['B'] = [1.0, 1, 2, 1, 2]
    tm.assert_frame_equal(result, expected)
    result = df.rolling(window='2s', min_periods=2).count()
    expected = df.copy()
    expected['B'] = [np.nan, np.nan, 2, np.nan, 2]
    tm.assert_frame_equal(result, expected)