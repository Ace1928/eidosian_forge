from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize(['values', 'window', 'min_periods', 'expected'], [[[20, 10, 10, np.inf, 1, 1, 2, 3], 3, 1, [np.nan, 50, 100 / 3, 0, 40.5, 0, 1 / 3, 1]], [[20, 10, 10, np.nan, 10, 1, 2, 3], 3, 1, [np.nan, 50, 100 / 3, 0, 0, 40.5, 73 / 3, 1]], [[np.nan, 5, 6, 7, 5, 5, 5], 3, 3, [np.nan] * 3 + [1, 1, 4 / 3, 0]], [[5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3], 3, 3, [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [1 / 3, 0]], [[5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3], 3, 3, [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [16 / 3, 0]], [[5, 7] * 4, 3, 3, [np.nan] * 2 + [4 / 3] * 6], [[5, 7, 5, np.nan, 7, 5, 7], 3, 2, [np.nan, 2, 4 / 3] + [2] * 3 + [4 / 3]]])
def test_rolling_var_same_value_count_logic(values, window, min_periods, expected):
    expected = Series(expected)
    sr = Series(values)
    result_var = sr.rolling(window, min_periods=min_periods).var()
    tm.assert_series_equal(result_var, expected)
    tm.assert_series_equal(expected == 0, result_var == 0)
    result_std = sr.rolling(window, min_periods=min_periods).std()
    tm.assert_series_equal(result_std, np.sqrt(expected))
    tm.assert_series_equal(expected == 0, result_std == 0)