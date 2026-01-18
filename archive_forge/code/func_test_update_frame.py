import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_update_frame(using_copy_on_write, warn_copy_on_write):
    df1 = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    df2 = DataFrame({'b': [100.0]}, index=[1])
    df1_orig = df1.copy()
    view = df1[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df1.update(df2)
    expected = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 100.0, 6.0]})
    tm.assert_frame_equal(df1, expected)
    if using_copy_on_write:
        tm.assert_frame_equal(view, df1_orig)
        assert np.shares_memory(get_array(df1, 'a'), get_array(view, 'a'))
        assert not np.shares_memory(get_array(df1, 'b'), get_array(view, 'b'))
    else:
        tm.assert_frame_equal(view, expected)