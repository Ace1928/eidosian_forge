import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('func, tz', [('tz_convert', 'Europe/Berlin'), ('tz_localize', None)])
def test_tz_convert_localize(using_copy_on_write, func, tz):
    ser = Series([1, 2], index=date_range(start='2014-08-01 09:00', freq='h', periods=2, tz=tz))
    ser_orig = ser.copy()
    ser2 = getattr(ser, func)('US/Central')
    if using_copy_on_write:
        assert np.shares_memory(ser.values, ser2.values)
    else:
        assert not np.shares_memory(ser.values, ser2.values)
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)