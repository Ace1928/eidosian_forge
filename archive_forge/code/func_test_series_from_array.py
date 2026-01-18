import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('fastpath', [False, True])
@pytest.mark.parametrize('dtype', [None, 'int64'])
@pytest.mark.parametrize('idx', [None, pd.RangeIndex(start=0, stop=3, step=1)])
@pytest.mark.parametrize('arr', [np.array([1, 2, 3], dtype='int64'), pd.array([1, 2, 3], dtype='Int64')])
def test_series_from_array(using_copy_on_write, idx, dtype, fastpath, arr):
    if idx is None or dtype is not None:
        fastpath = False
    msg = "The 'fastpath' keyword in pd.Series is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser = Series(arr, dtype=dtype, index=idx, fastpath=fastpath)
    ser_orig = ser.copy()
    data = getattr(arr, '_data', arr)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(ser), data)
    else:
        assert np.shares_memory(get_array(ser), data)
    arr[0] = 100
    if using_copy_on_write:
        tm.assert_series_equal(ser, ser_orig)
    else:
        expected = Series([100, 2, 3], dtype=dtype if dtype is not None else arr.dtype)
        tm.assert_series_equal(ser, expected)