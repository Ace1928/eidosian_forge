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
class TestSetitemCallable:

    def test_setitem_callable_key(self):
        ser = Series([1, 2, 3, 4], index=list('ABCD'))
        ser[lambda x: 'A'] = -1
        expected = Series([-1, 2, 3, 4], index=list('ABCD'))
        tm.assert_series_equal(ser, expected)

    def test_setitem_callable_other(self):
        inc = lambda x: x + 1
        ser = Series([1, 2, -1, 4], dtype=object)
        ser[ser < 0] = inc
        expected = Series([1, 2, inc, 4])
        tm.assert_series_equal(ser, expected)