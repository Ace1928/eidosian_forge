import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_index_na_position_key(self, sort_by_key):
    series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
    expected_series_first = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)
    index_sorted_series = series.sort_index(na_position='first', key=sort_by_key)
    tm.assert_series_equal(expected_series_first, index_sorted_series)
    expected_series_last = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)
    index_sorted_series = series.sort_index(na_position='last', key=sort_by_key)
    tm.assert_series_equal(expected_series_last, index_sorted_series)