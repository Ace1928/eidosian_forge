import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_subset_chained_single_block_row(using_copy_on_write, using_array_manager, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    df_orig = df.copy()
    subset = df[:].iloc[0].iloc[0:2]
    with tm.assert_cow_warning(warn_copy_on_write):
        subset.iloc[0] = 0
    if using_copy_on_write or using_array_manager:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.iloc[0, 0] == 0
    subset = df[:].iloc[0].iloc[0:2]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 0
    expected = Series([1, 4], index=['a', 'b'], name=0)
    if using_copy_on_write or using_array_manager:
        tm.assert_series_equal(subset, expected)
    else:
        assert subset.iloc[0] == 0