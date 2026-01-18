import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_droplevel_false_view(self, using_array_manager, using_copy_on_write, warn_copy_on_write):
    df = DataFrame([[1, 2, 3]], columns=Index(['a', 'b', 'c']))
    result = df.xs('a', axis=1, drop_level=False)
    assert np.shares_memory(result.iloc[:, 0]._values, df.iloc[:, 0]._values)
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 2
    if using_copy_on_write:
        expected = DataFrame({'a': [1]})
    else:
        expected = DataFrame({'a': [2]})
    tm.assert_frame_equal(result, expected)
    df = DataFrame([[1, 2.5, 'a']], columns=Index(['a', 'b', 'c']))
    result = df.xs('a', axis=1, drop_level=False)
    df.iloc[0, 0] = 2
    if using_copy_on_write:
        expected = DataFrame({'a': [1]})
    elif using_array_manager:
        expected = DataFrame({'a': [2]})
    else:
        expected = DataFrame({'a': [1]})
    tm.assert_frame_equal(result, expected)