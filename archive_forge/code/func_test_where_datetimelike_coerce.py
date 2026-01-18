import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['timedelta64[ns]', 'datetime64[ns]'])
def test_where_datetimelike_coerce(dtype):
    ser = Series([1, 2], dtype=dtype)
    expected = Series([10, 10])
    mask = np.array([False, False])
    msg = "Downcasting behavior in Series and DataFrame methods 'where'"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.where(mask, [10, 10])
    tm.assert_series_equal(rs, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.where(mask, 10)
    tm.assert_series_equal(rs, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.where(mask, 10.0)
    tm.assert_series_equal(rs, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.where(mask, [10.0, 10.0])
    tm.assert_series_equal(rs, expected)
    rs = ser.where(mask, [10.0, np.nan])
    expected = Series([10, np.nan], dtype='object')
    tm.assert_series_equal(rs, expected)