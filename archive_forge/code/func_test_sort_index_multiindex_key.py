import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_index_multiindex_key(self):
    mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
    s = Series([1, 2], mi)
    backwards = s.iloc[[1, 0]]
    result = s.sort_index(level='C', key=lambda x: -x)
    tm.assert_series_equal(s, result)
    result = s.sort_index(level='C', key=lambda x: x)
    tm.assert_series_equal(backwards, result)