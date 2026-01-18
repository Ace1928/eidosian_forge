import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_columnwise_no_op(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]})
    df_orig = df.copy()
    df2 = df.replace({'a': 10}, 100)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    df2.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df_orig)