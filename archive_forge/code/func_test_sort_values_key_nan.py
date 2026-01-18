import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_values_key_nan(self):
    series = Series(np.array([0, 5, np.nan, 3, 2, np.nan]))
    result = series.sort_values(axis=0)
    expected = series.iloc[[0, 4, 3, 1, 2, 5]]
    tm.assert_series_equal(result, expected)
    result = series.sort_values(axis=0, key=lambda x: x + 5)
    expected = series.iloc[[0, 4, 3, 1, 2, 5]]
    tm.assert_series_equal(result, expected)
    result = series.sort_values(axis=0, key=lambda x: -x, ascending=False)
    expected = series.iloc[[0, 4, 3, 1, 2, 5]]
    tm.assert_series_equal(result, expected)