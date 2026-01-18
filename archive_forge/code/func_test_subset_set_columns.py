import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['int64', 'float64'], ids=['single-block', 'mixed-block'])
def test_subset_set_columns(backend, using_copy_on_write, warn_copy_on_write, dtype):
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': np.array([7, 8, 9], dtype=dtype)})
    df_orig = df.copy()
    subset = df[1:3]
    if using_copy_on_write or warn_copy_on_write:
        subset[['a', 'c']] = 0
    else:
        with pd.option_context('chained_assignment', 'warn'):
            with tm.assert_produces_warning(SettingWithCopyWarning):
                subset[['a', 'c']] = 0
    subset._mgr._verify_integrity()
    if using_copy_on_write:
        assert all((subset._mgr._has_no_reference(i) for i in [0, 2]))
    expected = DataFrame({'a': [0, 0], 'b': [5, 6], 'c': [0, 0]}, index=range(1, 3))
    if dtype_backend == 'nullable':
        expected['a'] = expected['a'].astype('int64')
        expected['c'] = expected['c'].astype('int64')
    tm.assert_frame_equal(subset, expected)
    tm.assert_frame_equal(df, df_orig)