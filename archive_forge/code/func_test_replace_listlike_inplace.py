import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_listlike_inplace(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]})
    arr = get_array(df, 'a')
    df.replace([200, 2], [10, 11], inplace=True)
    assert np.shares_memory(get_array(df, 'a'), arr)
    view = df[:]
    df_orig = df.copy()
    with tm.assert_cow_warning(warn_copy_on_write):
        df.replace([200, 3], [10, 11], inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'a'), arr)
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(get_array(df, 'a'), arr)
        tm.assert_frame_equal(df, view)