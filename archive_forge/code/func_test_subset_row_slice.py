import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_subset_row_slice(backend, using_copy_on_write, warn_copy_on_write):
    _, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    subset = df[1:3]
    subset._mgr._verify_integrity()
    assert np.shares_memory(get_array(subset, 'a'), get_array(df, 'a'))
    if using_copy_on_write:
        subset.iloc[0, 0] = 0
        assert not np.shares_memory(get_array(subset, 'a'), get_array(df, 'a'))
    else:
        with tm.assert_cow_warning(warn_copy_on_write):
            subset.iloc[0, 0] = 0
    subset._mgr._verify_integrity()
    expected = DataFrame({'a': [0, 3], 'b': [5, 6], 'c': [0.2, 0.3]}, index=range(1, 3))
    tm.assert_frame_equal(subset, expected)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        df_orig.iloc[1, 0] = 0
        tm.assert_frame_equal(df, df_orig)