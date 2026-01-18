from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_setitem_empty_mask_dont_upcast_dt64():
    dti = date_range('2016-01-01', periods=3)
    ser = Series(dti)
    orig = ser.copy()
    mask = np.zeros(3, dtype=bool)
    ser[mask] = 'foo'
    assert ser.dtype == dti.dtype
    tm.assert_series_equal(ser, orig)
    ser.mask(mask, 'foo', inplace=True)
    assert ser.dtype == dti.dtype
    tm.assert_series_equal(ser, orig)