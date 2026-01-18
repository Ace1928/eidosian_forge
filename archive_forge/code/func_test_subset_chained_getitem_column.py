import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['int64', 'float64'], ids=['single-block', 'mixed-block'])
def test_subset_chained_getitem_column(backend, dtype, using_copy_on_write, warn_copy_on_write):
    dtype_backend, DataFrame, Series = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': np.array([7, 8, 9], dtype=dtype)})
    df_orig = df.copy()
    subset = df[:]['a'][0:2]
    df._clear_item_cache()
    with tm.assert_cow_warning(warn_copy_on_write):
        subset.iloc[0] = 0
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.iloc[0, 0] == 0
    subset = df[:]['a'][0:2]
    df._clear_item_cache()
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 0
    expected = Series([1, 2], name='a')
    if using_copy_on_write:
        tm.assert_series_equal(subset, expected)
    else:
        assert subset.iloc[0] == 0