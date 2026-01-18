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
def test_setitem_int_not_positional():
    ser = Series([1, 2, 3, 4], index=[1.1, 2.1, 3.0, 4.1])
    assert not ser.index._should_fallback_to_positional
    ser[3] = 10
    expected = Series([1, 2, 10, 4], index=ser.index)
    tm.assert_series_equal(ser, expected)
    ser[5] = 5
    expected = Series([1, 2, 10, 4, 5], index=[1.1, 2.1, 3.0, 4.1, 5.0])
    tm.assert_series_equal(ser, expected)
    ii = IntervalIndex.from_breaks(range(10))[::2]
    ser2 = Series(range(len(ii)), index=ii)
    exp_index = ii.astype(object).append(Index([4]))
    expected2 = Series([0, 1, 2, 3, 4, 9], index=exp_index)
    ser2[4] = 9
    tm.assert_series_equal(ser2, expected2)
    mi = MultiIndex.from_product([ser.index, ['A', 'B']])
    ser3 = Series(range(len(mi)), index=mi)
    expected3 = ser3.copy()
    expected3.loc[4] = 99
    ser3[4] = 99
    tm.assert_series_equal(ser3, expected3)