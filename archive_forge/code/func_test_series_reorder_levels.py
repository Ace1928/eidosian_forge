import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_series_reorder_levels(using_copy_on_write):
    index = MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1), (2, 2)], names=['one', 'two'])
    ser = Series([1, 2, 3, 4], index=index)
    ser_orig = ser.copy()
    ser2 = ser.reorder_levels(order=['two', 'one'])
    if using_copy_on_write:
        assert np.shares_memory(ser2.values, ser.values)
    else:
        assert not np.shares_memory(ser2.values, ser.values)
    ser2.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)