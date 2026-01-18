import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_regex_inplace_refs(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': ['aaa', 'bbb']})
    df_orig = df.copy()
    view = df[:]
    arr = get_array(df, 'a')
    with tm.assert_cow_warning(warn_copy_on_write):
        df.replace(to_replace='^a.*$', value='new', inplace=True, regex=True)
    if using_copy_on_write:
        assert not np.shares_memory(arr, get_array(df, 'a'))
        assert df._mgr._has_no_reference(0)
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(arr, get_array(df, 'a'))