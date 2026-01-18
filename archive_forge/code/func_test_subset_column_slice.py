import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['int64', 'float64'], ids=['single-block', 'mixed-block'])
def test_subset_column_slice(backend, using_copy_on_write, warn_copy_on_write, using_array_manager, dtype):
    dtype_backend, DataFrame, _ = backend
    single_block = (dtype == 'int64' and dtype_backend == 'numpy') and (not using_array_manager)
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': np.array([7, 8, 9], dtype=dtype)})
    df_orig = df.copy()
    subset = df.iloc[:, 1:]
    subset._mgr._verify_integrity()
    if using_copy_on_write:
        assert np.shares_memory(get_array(subset, 'b'), get_array(df, 'b'))
        subset.iloc[0, 0] = 0
        assert not np.shares_memory(get_array(subset, 'b'), get_array(df, 'b'))
    elif warn_copy_on_write:
        with tm.assert_cow_warning(single_block):
            subset.iloc[0, 0] = 0
    else:
        warn = SettingWithCopyWarning if single_block else None
        with pd.option_context('chained_assignment', 'warn'):
            with tm.assert_produces_warning(warn):
                subset.iloc[0, 0] = 0
    expected = DataFrame({'b': [0, 5, 6], 'c': np.array([7, 8, 9], dtype=dtype)})
    tm.assert_frame_equal(subset, expected)
    if not using_copy_on_write and (using_array_manager or single_block):
        df_orig.iloc[0, 1] = 0
        tm.assert_frame_equal(df, df_orig)
    else:
        tm.assert_frame_equal(df, df_orig)