import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_series_set_axis(using_copy_on_write):
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    ser2 = ser.set_axis(['a', 'b', 'c'], axis='index')
    if using_copy_on_write:
        assert np.shares_memory(ser, ser2)
    else:
        assert not np.shares_memory(ser, ser2)
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2, ser)
    tm.assert_series_equal(ser, ser_orig)