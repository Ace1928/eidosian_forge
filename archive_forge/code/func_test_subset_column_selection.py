import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_subset_column_selection(backend, using_copy_on_write):
    _, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    subset = df[['a', 'c']]
    if using_copy_on_write:
        assert np.shares_memory(get_array(subset, 'a'), get_array(df, 'a'))
        subset.iloc[0, 0] = 0
    else:
        assert not np.shares_memory(get_array(subset, 'a'), get_array(df, 'a'))
        subset.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(subset, 'a'), get_array(df, 'a'))
    expected = DataFrame({'a': [0, 2, 3], 'c': [0.1, 0.2, 0.3]})
    tm.assert_frame_equal(subset, expected)
    tm.assert_frame_equal(df, df_orig)