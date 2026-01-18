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
def test_setitem_boolean_nullable_int_types(self, any_numeric_ea_dtype):
    ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
    ser[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)
    expected = Series([5, 6, 2, 3], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(ser, expected)
    ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
    ser.loc[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(ser, expected)
    ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
    loc_ser = Series(range(4), dtype=any_numeric_ea_dtype)
    ser.loc[ser > 6] = loc_ser.loc[loc_ser > 1]
    tm.assert_series_equal(ser, expected)