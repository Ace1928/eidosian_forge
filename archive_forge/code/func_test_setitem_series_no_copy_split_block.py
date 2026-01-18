import numpy as np
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_setitem_series_no_copy_split_block(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': 1})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()
    df['b'] = rhs
    if using_copy_on_write:
        assert np.shares_memory(get_array(rhs), get_array(df, 'b'))
    df.iloc[0, 1] = 100
    tm.assert_series_equal(rhs, rhs_orig)