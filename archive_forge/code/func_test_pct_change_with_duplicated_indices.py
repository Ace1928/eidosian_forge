import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('fill_method', ['pad', 'ffill', None])
def test_pct_change_with_duplicated_indices(fill_method):
    data = DataFrame({0: [np.nan, 1, 2, 3, 9, 18], 1: [0, 1, np.nan, 3, 9, 18]}, index=['a', 'b'] * 3)
    warn = None if fill_method is None else FutureWarning
    msg = "The 'fill_method' keyword being not None and the 'limit' keyword in DataFrame.pct_change are deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        result = data.pct_change(fill_method=fill_method)
    if fill_method is None:
        second_column = [np.nan, np.inf, np.nan, np.nan, 2.0, 1.0]
    else:
        second_column = [np.nan, np.inf, 0.0, 2.0, 2.0, 1.0]
    expected = DataFrame({0: [np.nan, np.nan, 1.0, 0.5, 2.0, 1.0], 1: second_column}, index=['a', 'b'] * 3)
    tm.assert_frame_equal(result, expected)