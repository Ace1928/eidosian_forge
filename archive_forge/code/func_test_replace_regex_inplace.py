import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_regex_inplace(using_copy_on_write):
    df = DataFrame({'a': ['aaa', 'bbb']})
    arr = get_array(df, 'a')
    df.replace(to_replace='^a.*$', value='new', inplace=True, regex=True)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
    assert np.shares_memory(arr, get_array(df, 'a'))
    df_orig = df.copy()
    df2 = df.replace(to_replace='^b.*$', value='new', regex=True)
    tm.assert_frame_equal(df_orig, df)
    assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))