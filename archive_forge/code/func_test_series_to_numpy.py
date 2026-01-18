import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_series_to_numpy(using_copy_on_write):
    ser = Series([1, 2, 3], name='name')
    ser_orig = ser.copy()
    arr = ser.to_numpy()
    if using_copy_on_write:
        assert np.shares_memory(arr, get_array(ser, 'name'))
        assert arr.flags.writeable is False
        with pytest.raises(ValueError, match='read-only'):
            arr[0] = 0
        tm.assert_series_equal(ser, ser_orig)
        ser.iloc[0] = 0
        assert ser.values[0] == 0
    else:
        assert arr.flags.writeable is True
        arr[0] = 0
        assert ser.iloc[0] == 0
    ser = Series([1, 2, 3], name='name')
    arr = ser.to_numpy(copy=True)
    assert not np.shares_memory(arr, get_array(ser, 'name'))
    assert arr.flags.writeable is True
    ser = Series([1, 2, 3], name='name')
    arr = ser.to_numpy(dtype='float64')
    assert not np.shares_memory(arr, get_array(ser, 'name'))
    assert arr.flags.writeable is True