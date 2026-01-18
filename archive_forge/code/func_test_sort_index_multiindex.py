import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('level', ['A', 0])
def test_sort_index_multiindex(self, level):
    mi = MultiIndex.from_tuples([[2, 1, 3], [2, 1, 2], [1, 1, 1]], names=list('ABC'))
    df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mi)
    expected_mi = MultiIndex.from_tuples([[1, 1, 1], [2, 1, 2], [2, 1, 3]], names=list('ABC'))
    expected = DataFrame([[5, 6], [3, 4], [1, 2]], index=expected_mi)
    result = df.sort_index(level=level)
    tm.assert_frame_equal(result, expected)
    expected_mi = MultiIndex.from_tuples([[1, 1, 1], [2, 1, 3], [2, 1, 2]], names=list('ABC'))
    expected = DataFrame([[5, 6], [1, 2], [3, 4]], index=expected_mi)
    result = df.sort_index(level=level, sort_remaining=False)
    tm.assert_frame_equal(result, expected)