import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_reorder_levels_swaplevel_equivalence(self, multiindex_year_month_day_dataframe_random_data):
    ymd = multiindex_year_month_day_dataframe_random_data
    result = ymd.reorder_levels(['month', 'day', 'year'])
    expected = ymd.swaplevel(0, 1).swaplevel(1, 2)
    tm.assert_frame_equal(result, expected)
    result = ymd['A'].reorder_levels(['month', 'day', 'year'])
    expected = ymd['A'].swaplevel(0, 1).swaplevel(1, 2)
    tm.assert_series_equal(result, expected)
    result = ymd.T.reorder_levels(['month', 'day', 'year'], axis=1)
    expected = ymd.T.swaplevel(0, 1, axis=1).swaplevel(1, 2, axis=1)
    tm.assert_frame_equal(result, expected)
    with pytest.raises(TypeError, match='hierarchical axis'):
        ymd.reorder_levels([1, 2], axis=1)
    with pytest.raises(IndexError, match='Too many levels'):
        ymd.index.reorder_levels([1, 2, 3])