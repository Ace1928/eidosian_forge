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
def test_setitem_positional_float_into_int_coerces():
    ser = Series([1, 2, 3], index=['a', 'b', 'c'])
    warn_msg = 'Series.__setitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=warn_msg):
        ser[0] = 1.5
    expected = Series([1.5, 2, 3], index=['a', 'b', 'c'])
    tm.assert_series_equal(ser, expected)