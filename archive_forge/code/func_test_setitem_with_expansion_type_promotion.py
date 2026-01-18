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
def test_setitem_with_expansion_type_promotion(self):
    ser = Series(dtype=object)
    ser['a'] = Timestamp('2016-01-01')
    ser['b'] = 3.0
    ser['c'] = 'foo'
    expected = Series([Timestamp('2016-01-01'), 3.0, 'foo'], index=['a', 'b', 'c'])
    tm.assert_series_equal(ser, expected)