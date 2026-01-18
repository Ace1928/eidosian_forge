import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('expected_idx_non_na, ascending', [[[3, 4, 5, 0, 1, 8, 6, 9, 7, 10, 13, 14], [True, True]], [[0, 3, 4, 5, 1, 8, 6, 7, 10, 13, 14, 9], [True, False]], [[9, 7, 10, 13, 14, 6, 8, 1, 3, 4, 5, 0], [False, True]], [[7, 10, 13, 14, 9, 6, 8, 1, 0, 3, 4, 5], [False, False]]])
@pytest.mark.parametrize('na_position', ['first', 'last'])
def test_sort_values_stable_multicolumn_sort(self, expected_idx_non_na, ascending, na_position):
    df = DataFrame({'A': [1, 2, np.nan, 1, 1, 1, 6, 8, 4, 8, 8, np.nan, np.nan, 8, 8], 'B': [9, np.nan, 5, 2, 2, 2, 5, 4, 5, 3, 4, np.nan, np.nan, 4, 4]})
    expected_idx = [11, 12, 2] + expected_idx_non_na if na_position == 'first' else expected_idx_non_na + [2, 11, 12]
    expected = df.take(expected_idx)
    sorted_df = df.sort_values(['A', 'B'], ascending=ascending, na_position=na_position)
    tm.assert_frame_equal(sorted_df, expected)