import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_swapaxes_single_block(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['x', 'y', 'z'])
    df_orig = df.copy()
    msg = "'DataFrame.swapaxes' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df2 = df.swapaxes('index', 'columns')
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'x'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(df2, 'x'), get_array(df, 'a'))
    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, 'x'), get_array(df, 'a'))
    tm.assert_frame_equal(df, df_orig)