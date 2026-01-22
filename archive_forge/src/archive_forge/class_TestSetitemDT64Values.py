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
class TestSetitemDT64Values:

    def test_setitem_none_nan(self):
        series = Series(date_range('1/1/2000', periods=10))
        series[3] = None
        assert series[3] is NaT
        series[3:5] = None
        assert series[4] is NaT
        series[5] = np.nan
        assert series[5] is NaT
        series[5:7] = np.nan
        assert series[6] is NaT

    def test_setitem_multiindex_empty_slice(self):
        idx = MultiIndex.from_tuples([('a', 1), ('b', 2)])
        result = Series([1, 2], index=idx)
        expected = result.copy()
        result.loc[[]] = 0
        tm.assert_series_equal(result, expected)

    def test_setitem_with_string_index(self):
        ser = Series([1, 2, 3], index=['Date', 'b', 'other'], dtype=object)
        ser['Date'] = date.today()
        assert ser.Date == date.today()
        assert ser['Date'] == date.today()

    def test_setitem_tuple_with_datetimetz_values(self):
        arr = date_range('2017', periods=4, tz='US/Eastern')
        index = [(0, 1), (0, 2), (0, 3), (0, 4)]
        result = Series(arr, index=index)
        expected = result.copy()
        result[0, 1] = np.nan
        expected.iloc[0] = np.nan
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tz', ['US/Eastern', 'UTC', 'Asia/Tokyo'])
    def test_setitem_with_tz(self, tz, indexer_sli):
        orig = Series(date_range('2016-01-01', freq='h', periods=3, tz=tz))
        assert orig.dtype == f'datetime64[ns, {tz}]'
        exp = Series([Timestamp('2016-01-01 00:00', tz=tz), Timestamp('2011-01-01 00:00', tz=tz), Timestamp('2016-01-01 02:00', tz=tz)], dtype=orig.dtype)
        ser = orig.copy()
        indexer_sli(ser)[1] = Timestamp('2011-01-01', tz=tz)
        tm.assert_series_equal(ser, exp)
        vals = Series([Timestamp('2011-01-01', tz=tz), Timestamp('2012-01-01', tz=tz)], index=[1, 2], dtype=orig.dtype)
        assert vals.dtype == f'datetime64[ns, {tz}]'
        exp = Series([Timestamp('2016-01-01 00:00', tz=tz), Timestamp('2011-01-01 00:00', tz=tz), Timestamp('2012-01-01 00:00', tz=tz)], dtype=orig.dtype)
        ser = orig.copy()
        indexer_sli(ser)[[1, 2]] = vals
        tm.assert_series_equal(ser, exp)

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

    def test_object_series_setitem_dt64array_exact_match(self):
        ser = Series({'X': np.nan}, dtype=object)
        indexer = [True]
        value = np.array([4], dtype='M8[ns]')
        ser.iloc[indexer] = value
        expected = Series([value[0]], index=['X'], dtype=object)
        assert all((isinstance(x, np.datetime64) for x in expected.values))
        tm.assert_series_equal(ser, expected)