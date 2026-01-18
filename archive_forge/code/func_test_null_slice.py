import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('method', [lambda df: df[:], lambda df: df.loc[:, :], lambda df: df.loc[:], lambda df: df.iloc[:, :], lambda df: df.iloc[:]], ids=['getitem', 'loc', 'loc-rows', 'iloc', 'iloc-rows'])
def test_null_slice(backend, method, using_copy_on_write, warn_copy_on_write):
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    df_orig = df.copy()
    df2 = method(df)
    assert df2 is not df
    with tm.assert_cow_warning(warn_copy_on_write):
        df2.iloc[0, 0] = 0
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.iloc[0, 0] == 0