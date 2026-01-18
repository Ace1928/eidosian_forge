import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
def test_rolling_cov_diff_length():
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    result = s1.rolling(window=3, min_periods=2).cov(s2)
    expected = Series([None, None, 2.0])
    tm.assert_series_equal(result, expected)
    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.rolling(window=3, min_periods=2).cov(s2a)
    tm.assert_series_equal(result, expected)