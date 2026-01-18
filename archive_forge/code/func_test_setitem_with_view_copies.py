import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_setitem_with_view_copies(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 1})
    view = df[:]
    expected = df.copy()
    df['b'] = 100
    arr = get_array(df, 'a')
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(arr, get_array(df, 'a'))
        tm.assert_frame_equal(view, expected)