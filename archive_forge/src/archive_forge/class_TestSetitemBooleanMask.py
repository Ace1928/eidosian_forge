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
class TestSetitemBooleanMask:

    def test_setitem_mask_cast(self):
        ser = Series([1, 2], index=[1, 2], dtype='int64')
        ser[[True, False]] = Series([0], index=[1], dtype='int64')
        expected = Series([0, 2], index=[1, 2], dtype='int64')
        tm.assert_series_equal(ser, expected)

    def test_setitem_mask_align_and_promote(self):
        ts = Series(np.random.default_rng(2).standard_normal(100), index=np.arange(100, 0, -1)).round(5)
        mask = ts > 0
        left = ts.copy()
        right = ts[mask].copy().map(str)
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            left[mask] = right
        expected = ts.map(lambda t: str(t) if t > 0 else t)
        tm.assert_series_equal(left, expected)

    def test_setitem_mask_promote_strs(self):
        ser = Series([0, 1, 2, 0])
        mask = ser > 0
        ser2 = ser[mask].map(str)
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            ser[mask] = ser2
        expected = Series([0, '1', '2', 0])
        tm.assert_series_equal(ser, expected)

    def test_setitem_mask_promote(self):
        ser = Series([0, 'foo', 'bar', 0])
        mask = Series([False, True, True, False])
        ser2 = ser[mask]
        ser[mask] = ser2
        expected = Series([0, 'foo', 'bar', 0])
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean(self, string_series):
        mask = string_series > string_series.median()
        result = string_series.copy()
        result[mask] = string_series * 2
        expected = string_series * 2
        tm.assert_series_equal(result[mask], expected[mask])
        result = string_series.copy()
        result[mask] = (string_series * 2)[0:5]
        expected = (string_series * 2)[0:5].reindex_like(string_series)
        expected[-mask] = string_series[mask]
        tm.assert_series_equal(result[mask], expected[mask])

    def test_setitem_boolean_corner(self, datetime_series):
        ts = datetime_series
        mask_shifted = ts.shift(1, freq=BDay()) > ts.median()
        msg = 'Unalignable boolean Series provided as indexer \\(index of the boolean Series and of the indexed object do not match'
        with pytest.raises(IndexingError, match=msg):
            ts[mask_shifted] = 1
        with pytest.raises(IndexingError, match=msg):
            ts.loc[mask_shifted] = 1

    def test_setitem_boolean_different_order(self, string_series):
        ordered = string_series.sort_values()
        copy = string_series.copy()
        copy[ordered > 0] = 0
        expected = string_series.copy()
        expected[expected > 0] = 0
        tm.assert_series_equal(copy, expected)

    @pytest.mark.parametrize('func', [list, np.array, Series])
    def test_setitem_boolean_python_list(self, func):
        ser = Series([None, 'b', None])
        mask = func([True, False, True])
        ser[mask] = ['a', 'c']
        expected = Series(['a', 'b', 'c'])
        tm.assert_series_equal(ser, expected)

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

    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self):
        ser = Series([None] * 10)
        mask = [False] * 3 + [True] * 5 + [False] * 2
        ser[mask] = range(5)
        result = ser
        expected = Series([None] * 3 + list(range(5)) + [None] * 2, dtype=object)
        tm.assert_series_equal(result, expected)

    def test_setitem_nan_with_bool(self):
        result = Series([True, False, True])
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            result[0] = np.nan
        expected = Series([np.nan, False, True], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_setitem_mask_smallint_upcast(self):
        orig = Series([1, 2, 3], dtype='int8')
        alt = np.array([999, 1000, 1001], dtype=np.int64)
        mask = np.array([True, False, True])
        ser = orig.copy()
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            ser[mask] = Series(alt)
        expected = Series([999, 2, 1001])
        tm.assert_series_equal(ser, expected)
        ser2 = orig.copy()
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            ser2.mask(mask, alt, inplace=True)
        tm.assert_series_equal(ser2, expected)
        ser3 = orig.copy()
        res = ser3.where(~mask, Series(alt))
        tm.assert_series_equal(res, expected)

    def test_setitem_mask_smallint_no_upcast(self):
        orig = Series([1, 2, 3], dtype='uint8')
        alt = Series([245, 1000, 246], dtype=np.int64)
        mask = np.array([True, False, True])
        ser = orig.copy()
        ser[mask] = alt
        expected = Series([245, 2, 246], dtype='uint8')
        tm.assert_series_equal(ser, expected)
        ser2 = orig.copy()
        ser2.mask(mask, alt, inplace=True)
        tm.assert_series_equal(ser2, expected)
        ser3 = orig.copy()
        res = ser3.where(~mask, alt)
        tm.assert_series_equal(res, expected, check_dtype=False)