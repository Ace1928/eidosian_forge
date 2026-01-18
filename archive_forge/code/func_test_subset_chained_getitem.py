import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('method', [lambda df: df[['a', 'b']][0:2], lambda df: df[0:2][['a', 'b']], lambda df: df[['a', 'b']].iloc[0:2], lambda df: df[['a', 'b']].loc[0:1], lambda df: df[0:2].iloc[:, 0:2], lambda df: df[0:2].loc[:, 'a':'b']], ids=['row-getitem-slice', 'column-getitem', 'row-iloc-slice', 'row-loc-slice', 'column-iloc-slice', 'column-loc-slice'])
@pytest.mark.parametrize('dtype', ['int64', 'float64'], ids=['single-block', 'mixed-block'])
def test_subset_chained_getitem(request, backend, method, dtype, using_copy_on_write, using_array_manager, warn_copy_on_write):
    _, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': np.array([7, 8, 9], dtype=dtype)})
    df_orig = df.copy()
    test_callspec = request.node.callspec.id
    if not using_array_manager:
        subset_is_view = test_callspec in ('numpy-single-block-column-iloc-slice', 'numpy-single-block-column-loc-slice')
    else:
        subset_is_view = test_callspec.endswith(('column-iloc-slice', 'column-loc-slice'))
    subset = method(df)
    with tm.assert_cow_warning(warn_copy_on_write and subset_is_view):
        subset.iloc[0, 0] = 0
    if using_copy_on_write or not subset_is_view:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.iloc[0, 0] == 0
    subset = method(df)
    with tm.assert_cow_warning(warn_copy_on_write and subset_is_view):
        df.iloc[0, 0] = 0
    expected = DataFrame({'a': [1, 2], 'b': [4, 5]})
    if using_copy_on_write or not subset_is_view:
        tm.assert_frame_equal(subset, expected)
    else:
        assert subset.iloc[0, 0] == 0