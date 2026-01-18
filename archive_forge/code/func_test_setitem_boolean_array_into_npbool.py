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
def test_setitem_boolean_array_into_npbool(self):
    ser = Series([True, False, True])
    values = ser._values
    arr = array([True, False, None])
    ser[:2] = arr[:2]
    assert ser._values is values
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser[1:] = arr[1:]
    expected = Series(arr)
    tm.assert_series_equal(ser, expected)