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
def test_setitem_mask_smallint_no_upcast(self):
    orig = Series([1, 2, 3], dtype='uint8')
    alt = Series([245, 1000, 246], dtype=np.int64)
    mask = np.array([True, False, True])
    ser = orig.copy()
    ser[mask] = alt
    expected = Series([245, 2, 246], dtype='uint8')
    tm.assert_series_equal(ser, expected)
    ser2 = orig.copy()
    ser2.mask(mask, alt, inplace=True)
    tm.assert_series_equal(ser2, expected)
    ser3 = orig.copy()
    res = ser3.where(~mask, alt)
    tm.assert_series_equal(res, expected, check_dtype=False)