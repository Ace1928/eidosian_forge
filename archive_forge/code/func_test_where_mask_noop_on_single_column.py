import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype, val', [('int64', 10.5), ('Int64', 10)])
@pytest.mark.parametrize('func', [lambda df, val: df.where(df < 0, val), lambda df, val: df.mask(df >= 0, val)])
def test_where_mask_noop_on_single_column(using_copy_on_write, dtype, val, func):
    df = DataFrame({'a': [1, 2, 3], 'b': [-4, -5, -6]}, dtype=dtype)
    df_orig = df.copy()
    result = func(df, val)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'b'), get_array(result, 'b'))
        assert not np.shares_memory(get_array(df, 'a'), get_array(result, 'a'))
    else:
        assert not np.shares_memory(get_array(df, 'b'), get_array(result, 'b'))
    result.iloc[0, 1] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'b'), get_array(result, 'b'))
    tm.assert_frame_equal(df, df_orig)