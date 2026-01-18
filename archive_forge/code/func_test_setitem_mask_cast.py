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
def test_setitem_mask_cast(self):
    ser = Series([1, 2], index=[1, 2], dtype='int64')
    ser[[True, False]] = Series([0], index=[1], dtype='int64')
    expected = Series([0, 2], index=[1, 2], dtype='int64')
    tm.assert_series_equal(ser, expected)