from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('level', [0, 'first', 1, 'second'])
def test_unique_level(idx, level):
    result = idx.unique(level=level)
    expected = idx.get_level_values(level).unique()
    tm.assert_index_equal(result, expected)
    mi = MultiIndex.from_arrays([[1, 3, 2, 4], [1, 3, 2, 5]], names=['first', 'second'])
    result = mi.unique(level=level)
    expected = mi.get_level_values(level)
    tm.assert_index_equal(result, expected)
    mi = MultiIndex.from_arrays([[], []], names=['first', 'second'])
    result = mi.unique(level=level)
    expected = mi.get_level_values(level)
    tm.assert_index_equal(result, expected)