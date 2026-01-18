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
def test_setitem_with_tz_dst(self, indexer_sli):
    tz = 'US/Eastern'
    orig = Series(date_range('2016-11-06', freq='h', periods=3, tz=tz))
    assert orig.dtype == f'datetime64[ns, {tz}]'
    exp = Series([Timestamp('2016-11-06 00:00-04:00', tz=tz), Timestamp('2011-01-01 00:00-05:00', tz=tz), Timestamp('2016-11-06 01:00-05:00', tz=tz)], dtype=orig.dtype)
    ser = orig.copy()
    indexer_sli(ser)[1] = Timestamp('2011-01-01', tz=tz)
    tm.assert_series_equal(ser, exp)
    vals = Series([Timestamp('2011-01-01', tz=tz), Timestamp('2012-01-01', tz=tz)], index=[1, 2], dtype=orig.dtype)
    assert vals.dtype == f'datetime64[ns, {tz}]'
    exp = Series([Timestamp('2016-11-06 00:00', tz=tz), Timestamp('2011-01-01 00:00', tz=tz), Timestamp('2012-01-01 00:00', tz=tz)], dtype=orig.dtype)
    ser = orig.copy()
    indexer_sli(ser)[[1, 2]] = vals
    tm.assert_series_equal(ser, exp)