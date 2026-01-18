import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('val', [100, 'a'])
@pytest.mark.parametrize('indexer_func, indexer', [(tm.loc, (0, 'a')), (tm.iloc, (0, 0)), (tm.loc, ([0], 'a')), (tm.iloc, ([0], 0)), (tm.loc, (slice(None), 'a')), (tm.iloc, (slice(None), 0))])
@pytest.mark.parametrize('col', [[0.1, 0.2, 0.3], [7, 8, 9]], ids=['mixed-block', 'single-block'])
def test_set_value_copy_only_necessary_column(using_copy_on_write, warn_copy_on_write, indexer_func, indexer, val, col):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': col})
    df_orig = df.copy()
    view = df[:]
    if val == 'a' and (not warn_copy_on_write):
        with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype is deprecated'):
            indexer_func(df)[indexer] = val
    if val == 'a' and warn_copy_on_write:
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype|Setting a value on a view'):
            indexer_func(df)[indexer] = val
    else:
        with tm.assert_cow_warning(warn_copy_on_write and val == 100):
            indexer_func(df)[indexer] = val
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'b'), get_array(view, 'b'))
        assert not np.shares_memory(get_array(df, 'a'), get_array(view, 'a'))
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(get_array(df, 'c'), get_array(view, 'c'))
        if val == 'a':
            assert not np.shares_memory(get_array(df, 'a'), get_array(view, 'a'))
        else:
            assert np.shares_memory(get_array(df, 'a'), get_array(view, 'a'))