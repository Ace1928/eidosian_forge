import numpy as np
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_set_column_with_array():
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    arr = np.array([1, 2, 3], dtype='int64')
    df['c'] = arr
    assert not np.shares_memory(get_array(df, 'c'), arr)
    arr[0] = 0
    tm.assert_series_equal(df['c'], Series([1, 2, 3], name='c'))