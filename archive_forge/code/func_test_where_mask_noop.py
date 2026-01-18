import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['int64', 'Int64'])
@pytest.mark.parametrize('func', [lambda ser: ser.where(ser > 0, 10), lambda ser: ser.mask(ser <= 0, 10)])
def test_where_mask_noop(using_copy_on_write, dtype, func):
    ser = Series([1, 2, 3], dtype=dtype)
    ser_orig = ser.copy()
    result = func(ser)
    if using_copy_on_write:
        assert np.shares_memory(get_array(ser), get_array(result))
    else:
        assert not np.shares_memory(get_array(ser), get_array(result))
    result.iloc[0] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(ser), get_array(result))
    tm.assert_series_equal(ser, ser_orig)