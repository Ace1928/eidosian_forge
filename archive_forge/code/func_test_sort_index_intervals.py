import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_index_intervals(self):
    s = Series([np.nan, 1, 2, 3], IntervalIndex.from_arrays([0, 1, 2, 3], [1, 2, 3, 4]))
    result = s.sort_index()
    expected = s
    tm.assert_series_equal(result, expected)
    result = s.sort_index(ascending=False)
    expected = Series([3, 2, 1, np.nan], IntervalIndex.from_arrays([3, 2, 1, 0], [4, 3, 2, 1]))
    tm.assert_series_equal(result, expected)