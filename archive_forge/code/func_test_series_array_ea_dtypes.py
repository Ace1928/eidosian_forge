import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_series_array_ea_dtypes(using_copy_on_write):
    ser = Series([1, 2, 3], dtype='Int64')
    arr = np.asarray(ser, dtype='int64')
    assert np.shares_memory(arr, get_array(ser))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True
    arr = np.asarray(ser)
    assert np.shares_memory(arr, get_array(ser))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True