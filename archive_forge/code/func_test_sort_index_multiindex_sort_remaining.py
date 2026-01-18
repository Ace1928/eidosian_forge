import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ascending', [True, False])
def test_sort_index_multiindex_sort_remaining(self, ascending):
    df = DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}, index=MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'x'), ('b', 'y'), ('c', 'x')]))
    result = df.sort_index(level=1, sort_remaining=False, ascending=ascending)
    if ascending:
        expected = DataFrame({'A': [1, 3, 5, 2, 4], 'B': [10, 30, 50, 20, 40]}, index=MultiIndex.from_tuples([('a', 'x'), ('b', 'x'), ('c', 'x'), ('a', 'y'), ('b', 'y')]))
    else:
        expected = DataFrame({'A': [2, 4, 1, 3, 5], 'B': [20, 40, 10, 30, 50]}, index=MultiIndex.from_tuples([('a', 'y'), ('b', 'y'), ('a', 'x'), ('b', 'x'), ('c', 'x')]))
    tm.assert_frame_equal(result, expected)