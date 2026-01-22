from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
class TestTimedeltaArraylikeMulDivOps:

    def test_td64arr_mul_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx = tm.box_expected(idx, box_with_array)
        result = idx * 1
        tm.assert_equal(result, idx)
        result = 1 * idx
        tm.assert_equal(result, idx)

    def test_td64arr_mul_tdlike_scalar_raises(self, two_hours, box_with_array):
        rng = timedelta_range('1 days', '10 days', name='foo')
        rng = tm.box_expected(rng, box_with_array)
        msg = 'argument must be an integer|cannot use operands with types dtype'
        with pytest.raises(TypeError, match=msg):
            rng * two_hours

    def test_tdi_mul_int_array_zerodim(self, box_with_array):
        rng5 = np.arange(5, dtype='int64')
        idx = TimedeltaIndex(rng5)
        expected = TimedeltaIndex(rng5 * 5)
        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = idx * np.array(5, dtype='int64')
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_array(self, box_with_array):
        rng5 = np.arange(5, dtype='int64')
        idx = TimedeltaIndex(rng5)
        expected = TimedeltaIndex(rng5 ** 2)
        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = idx * rng5
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_series(self, box_with_array):
        box = box_with_array
        xbox = Series if box in [Index, tm.to_array, pd.array] else box
        idx = TimedeltaIndex(np.arange(5, dtype='int64'))
        expected = TimedeltaIndex(np.arange(5, dtype='int64') ** 2)
        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, xbox)
        result = idx * Series(np.arange(5, dtype='int64'))
        tm.assert_equal(result, expected)

    def test_tdi_mul_float_series(self, box_with_array):
        box = box_with_array
        xbox = Series if box in [Index, tm.to_array, pd.array] else box
        idx = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx = tm.box_expected(idx, box)
        rng5f = np.arange(5, dtype='float64')
        expected = TimedeltaIndex(rng5f * (rng5f + 1.0))
        expected = tm.box_expected(expected, xbox)
        result = idx * Series(rng5f + 1.0)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('other', [np.arange(1, 11), Index(np.arange(1, 11), np.int64), Index(range(1, 11), np.uint64), Index(range(1, 11), np.float64), pd.RangeIndex(1, 11)], ids=lambda x: type(x).__name__)
    def test_tdi_rmul_arraylike(self, other, box_with_array):
        box = box_with_array
        tdi = TimedeltaIndex(['1 Day'] * 10)
        expected = timedelta_range('1 days', '10 days')._with_freq(None)
        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(tdi, other)
        expected = tm.box_expected(expected, xbox)
        result = other * tdi
        tm.assert_equal(result, expected)
        commute = tdi * other
        tm.assert_equal(commute, expected)

    def test_td64arr_div_nat_invalid(self, box_with_array):
        rng = timedelta_range('1 days', '10 days', name='foo')
        rng = tm.box_expected(rng, box_with_array)
        with pytest.raises(TypeError, match='unsupported operand type'):
            rng / NaT
        with pytest.raises(TypeError, match='Cannot divide NaTType by'):
            NaT / rng
        dt64nat = np.datetime64('NaT', 'ns')
        msg = '|'.join(["ufunc '(true_divide|divide)' cannot use operands", 'cannot perform __r?truediv__', 'Cannot divide datetime64 by TimedeltaArray'])
        with pytest.raises(TypeError, match=msg):
            rng / dt64nat
        with pytest.raises(TypeError, match=msg):
            dt64nat / rng

    def test_td64arr_div_td64nat(self, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        rng = timedelta_range('1 days', '10 days')
        rng = tm.box_expected(rng, box)
        other = np.timedelta64('NaT')
        expected = np.array([np.nan] * 10)
        expected = tm.box_expected(expected, xbox)
        result = rng / other
        tm.assert_equal(result, expected)
        result = other / rng
        tm.assert_equal(result, expected)

    def test_td64arr_div_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx = tm.box_expected(idx, box_with_array)
        result = idx / 1
        tm.assert_equal(result, idx)
        with pytest.raises(TypeError, match='Cannot divide'):
            1 / idx

    def test_td64arr_div_tdlike_scalar(self, two_hours, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        rng = timedelta_range('1 days', '10 days', name='foo')
        expected = Index((np.arange(10) + 1) * 12, dtype=np.float64, name='foo')
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)
        result = rng / two_hours
        tm.assert_equal(result, expected)
        result = two_hours / rng
        expected = 1 / expected
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('m', [1, 3, 10])
    @pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
    def test_td64arr_div_td64_scalar(self, m, unit, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        ser = Series([Timedelta(days=59)] * 3)
        ser[2] = np.nan
        flat = ser
        ser = tm.box_expected(ser, box)
        expected = Series([x / np.timedelta64(m, unit) for x in flat])
        expected = tm.box_expected(expected, xbox)
        result = ser / np.timedelta64(m, unit)
        tm.assert_equal(result, expected)
        expected = Series([Timedelta(np.timedelta64(m, unit)) / x for x in flat])
        expected = tm.box_expected(expected, xbox)
        result = np.timedelta64(m, unit) / ser
        tm.assert_equal(result, expected)

    def test_td64arr_div_tdlike_scalar_with_nat(self, two_hours, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        rng = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        expected = Index([12, np.nan, 24], dtype=np.float64, name='foo')
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)
        result = rng / two_hours
        tm.assert_equal(result, expected)
        result = two_hours / rng
        expected = 1 / expected
        tm.assert_equal(result, expected)

    def test_td64arr_div_td64_ndarray(self, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        rng = TimedeltaIndex(['1 days', NaT, '2 days'])
        expected = Index([12, np.nan, 24], dtype=np.float64)
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)
        other = np.array([2, 4, 2], dtype='m8[h]')
        result = rng / other
        tm.assert_equal(result, expected)
        result = rng / tm.box_expected(other, box)
        tm.assert_equal(result, expected)
        result = rng / other.astype(object)
        tm.assert_equal(result, expected.astype(object))
        result = rng / list(other)
        tm.assert_equal(result, expected)
        expected = 1 / expected
        result = other / rng
        tm.assert_equal(result, expected)
        result = tm.box_expected(other, box) / rng
        tm.assert_equal(result, expected)
        result = other.astype(object) / rng
        tm.assert_equal(result, expected)
        result = list(other) / rng
        tm.assert_equal(result, expected)

    def test_tdarr_div_length_mismatch(self, box_with_array):
        rng = TimedeltaIndex(['1 days', NaT, '2 days'])
        mismatched = [1, 2, 3, 4]
        rng = tm.box_expected(rng, box_with_array)
        msg = 'Cannot divide vectors|Unable to coerce to Series'
        for obj in [mismatched, mismatched[:2]]:
            for other in [obj, np.array(obj), Index(obj)]:
                with pytest.raises(ValueError, match=msg):
                    rng / other
                with pytest.raises(ValueError, match=msg):
                    other / rng

    def test_td64_div_object_mixed_result(self, box_with_array):
        orig = timedelta_range('1 Day', periods=3).insert(1, NaT)
        tdi = tm.box_expected(orig, box_with_array, transpose=False)
        other = np.array([orig[0], 1.5, 2.0, orig[2]], dtype=object)
        other = tm.box_expected(other, box_with_array, transpose=False)
        res = tdi / other
        expected = Index([1.0, np.timedelta64('NaT', 'ns'), orig[0], 1.5], dtype=object)
        expected = tm.box_expected(expected, box_with_array, transpose=False)
        if isinstance(expected, NumpyExtensionArray):
            expected = expected.to_numpy()
        tm.assert_equal(res, expected)
        if box_with_array is DataFrame:
            assert isinstance(res.iloc[1, 0], np.timedelta64)
        res = tdi // other
        expected = Index([1, np.timedelta64('NaT', 'ns'), orig[0], 1], dtype=object)
        expected = tm.box_expected(expected, box_with_array, transpose=False)
        if isinstance(expected, NumpyExtensionArray):
            expected = expected.to_numpy()
        tm.assert_equal(res, expected)
        if box_with_array is DataFrame:
            assert isinstance(res.iloc[1, 0], np.timedelta64)

    def test_td64arr_floordiv_td64arr_with_nat(self, box_with_array, using_array_manager):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        left = Series([1000, 222330, 30], dtype='timedelta64[ns]')
        right = Series([1000, 222330, None], dtype='timedelta64[ns]')
        left = tm.box_expected(left, box)
        right = tm.box_expected(right, box)
        expected = np.array([1.0, 1.0, np.nan], dtype=np.float64)
        expected = tm.box_expected(expected, xbox)
        if box is DataFrame and using_array_manager:
            expected[[0, 1]] = expected[[0, 1]].astype('int64')
        with tm.maybe_produces_warning(RuntimeWarning, box is pd.array, check_stacklevel=False):
            result = left // right
        tm.assert_equal(result, expected)
        with tm.maybe_produces_warning(RuntimeWarning, box is pd.array, check_stacklevel=False):
            result = np.asarray(left) // right
        tm.assert_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')
    def test_td64arr_floordiv_tdscalar(self, box_with_array, scalar_td):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        td = Timedelta('5m3s')
        td1 = Series([td, td, NaT], dtype='m8[ns]')
        td1 = tm.box_expected(td1, box, transpose=False)
        expected = Series([0, 0, np.nan])
        expected = tm.box_expected(expected, xbox, transpose=False)
        result = td1 // scalar_td
        tm.assert_equal(result, expected)
        expected = Series([2, 2, np.nan])
        expected = tm.box_expected(expected, xbox, transpose=False)
        result = scalar_td // td1
        tm.assert_equal(result, expected)
        result = td1.__rfloordiv__(scalar_td)
        tm.assert_equal(result, expected)

    def test_td64arr_floordiv_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx = tm.box_expected(idx, box_with_array)
        result = idx // 1
        tm.assert_equal(result, idx)
        pattern = 'floor_divide cannot use operands|Cannot divide int by Timedelta*'
        with pytest.raises(TypeError, match=pattern):
            1 // idx

    def test_td64arr_mod_tdscalar(self, box_with_array, three_days):
        tdi = timedelta_range('1 Day', '9 days')
        tdarr = tm.box_expected(tdi, box_with_array)
        expected = TimedeltaIndex(['1 Day', '2 Days', '0 Days'] * 3)
        expected = tm.box_expected(expected, box_with_array)
        result = tdarr % three_days
        tm.assert_equal(result, expected)
        warn = None
        if box_with_array is DataFrame and isinstance(three_days, pd.DateOffset):
            warn = PerformanceWarning
            expected = expected.astype(object)
        with tm.assert_produces_warning(warn):
            result = divmod(tdarr, three_days)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // three_days)

    def test_td64arr_mod_int(self, box_with_array):
        tdi = timedelta_range('1 ns', '10 ns', periods=10)
        tdarr = tm.box_expected(tdi, box_with_array)
        expected = TimedeltaIndex(['1 ns', '0 ns'] * 5)
        expected = tm.box_expected(expected, box_with_array)
        result = tdarr % 2
        tm.assert_equal(result, expected)
        msg = 'Cannot divide int by'
        with pytest.raises(TypeError, match=msg):
            2 % tdarr
        result = divmod(tdarr, 2)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // 2)

    def test_td64arr_rmod_tdscalar(self, box_with_array, three_days):
        tdi = timedelta_range('1 Day', '9 days')
        tdarr = tm.box_expected(tdi, box_with_array)
        expected = ['0 Days', '1 Day', '0 Days'] + ['3 Days'] * 6
        expected = TimedeltaIndex(expected)
        expected = tm.box_expected(expected, box_with_array)
        result = three_days % tdarr
        tm.assert_equal(result, expected)
        result = divmod(three_days, tdarr)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], three_days // tdarr)

    def test_td64arr_mul_tdscalar_invalid(self, box_with_array, scalar_td):
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan
        td1 = tm.box_expected(td1, box_with_array)
        pattern = 'operate|unsupported|cannot|not supported'
        with pytest.raises(TypeError, match=pattern):
            td1 * scalar_td
        with pytest.raises(TypeError, match=pattern):
            scalar_td * td1

    def test_td64arr_mul_too_short_raises(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx = tm.box_expected(idx, box_with_array)
        msg = '|'.join(['cannot use operands with types dtype', 'Cannot multiply with unequal lengths', 'Unable to coerce to Series'])
        with pytest.raises(TypeError, match=msg):
            idx * idx[:3]
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    def test_td64arr_mul_td64arr_raises(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx = tm.box_expected(idx, box_with_array)
        msg = 'cannot use operands with types dtype'
        with pytest.raises(TypeError, match=msg):
            idx * idx

    def test_td64arr_mul_numeric_scalar(self, box_with_array, one):
        tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        expected = Series(['-59 Days', '-59 Days', 'NaT'], dtype='timedelta64[ns]')
        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = tdser * -one
        tm.assert_equal(result, expected)
        result = -one * tdser
        tm.assert_equal(result, expected)
        expected = Series(['118 Days', '118 Days', 'NaT'], dtype='timedelta64[ns]')
        expected = tm.box_expected(expected, box_with_array)
        result = tdser * (2 * one)
        tm.assert_equal(result, expected)
        result = 2 * one * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('two', [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_div_numeric_scalar(self, box_with_array, two):
        tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        expected = Series(['29.5D', '29.5D', 'NaT'], dtype='timedelta64[ns]')
        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = tdser / two
        tm.assert_equal(result, expected)
        with pytest.raises(TypeError, match='Cannot divide'):
            two / tdser

    @pytest.mark.parametrize('two', [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_floordiv_numeric_scalar(self, box_with_array, two):
        tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        expected = Series(['29.5D', '29.5D', 'NaT'], dtype='timedelta64[ns]')
        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = tdser // two
        tm.assert_equal(result, expected)
        with pytest.raises(TypeError, match='Cannot divide'):
            two // tdser

    @pytest.mark.parametrize('vector', [np.array([20, 30, 40]), Index([20, 30, 40]), Series([20, 30, 40])], ids=lambda x: type(x).__name__)
    def test_td64arr_rmul_numeric_array(self, box_with_array, vector, any_real_numpy_dtype):
        tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        vector = vector.astype(any_real_numpy_dtype)
        expected = Series(['1180 Days', '1770 Days', 'NaT'], dtype='timedelta64[ns]')
        tdser = tm.box_expected(tdser, box_with_array)
        xbox = get_upcast_box(tdser, vector)
        expected = tm.box_expected(expected, xbox)
        result = tdser * vector
        tm.assert_equal(result, expected)
        result = vector * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('vector', [np.array([20, 30, 40]), Index([20, 30, 40]), Series([20, 30, 40])], ids=lambda x: type(x).__name__)
    def test_td64arr_div_numeric_array(self, box_with_array, vector, any_real_numpy_dtype):
        tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        vector = vector.astype(any_real_numpy_dtype)
        expected = Series(['2.95D', '1D 23h 12m', 'NaT'], dtype='timedelta64[ns]')
        tdser = tm.box_expected(tdser, box_with_array)
        xbox = get_upcast_box(tdser, vector)
        expected = tm.box_expected(expected, xbox)
        result = tdser / vector
        tm.assert_equal(result, expected)
        pattern = '|'.join(["true_divide'? cannot use operands", 'cannot perform __div__', 'cannot perform __truediv__', 'unsupported operand', 'Cannot divide', "ufunc 'divide' cannot use operands with types"])
        with pytest.raises(TypeError, match=pattern):
            vector / tdser
        result = tdser / vector.astype(object)
        if box_with_array is DataFrame:
            expected = [tdser.iloc[0, n] / vector[n] for n in range(len(vector))]
            expected = tm.box_expected(expected, xbox).astype(object)
            msg = "The 'downcast' keyword in fillna"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                expected[2] = expected[2].fillna(np.timedelta64('NaT', 'ns'), downcast=False)
        else:
            expected = [tdser[n] / vector[n] for n in range(len(tdser))]
            expected = [x if x is not NaT else np.timedelta64('NaT', 'ns') for x in expected]
            if xbox is tm.to_array:
                expected = tm.to_array(expected).astype(object)
            else:
                expected = xbox(expected, dtype=object)
        tm.assert_equal(result, expected)
        with pytest.raises(TypeError, match=pattern):
            vector.astype(object) / tdser

    def test_td64arr_mul_int_series(self, box_with_array, names):
        box = box_with_array
        exname = get_expected_name(box, names)
        tdi = TimedeltaIndex(['0days', '1day', '2days', '3days', '4days'], name=names[0])
        ser = Series([0, 1, 2, 3, 4], dtype=np.int64, name=names[1])
        expected = Series(['0days', '1day', '4days', '9days', '16days'], dtype='timedelta64[ns]', name=exname)
        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(tdi, ser)
        expected = tm.box_expected(expected, xbox)
        result = ser * tdi
        tm.assert_equal(result, expected)
        result = tdi * ser
        tm.assert_equal(result, expected)

    def test_float_series_rdiv_td64arr(self, box_with_array, names):
        box = box_with_array
        tdi = TimedeltaIndex(['0days', '1day', '2days', '3days', '4days'], name=names[0])
        ser = Series([1.5, 3, 4.5, 6, 7.5], dtype=np.float64, name=names[1])
        xname = names[2] if box not in [tm.to_array, pd.array] else names[1]
        expected = Series([tdi[n] / ser[n] for n in range(len(ser))], dtype='timedelta64[ns]', name=xname)
        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(tdi, ser)
        expected = tm.box_expected(expected, xbox)
        result = ser.__rtruediv__(tdi)
        if box is DataFrame:
            assert result is NotImplemented
        else:
            tm.assert_equal(result, expected)

    def test_td64arr_all_nat_div_object_dtype_numeric(self, box_with_array):
        tdi = TimedeltaIndex([NaT, NaT])
        left = tm.box_expected(tdi, box_with_array)
        right = np.array([2, 2.0], dtype=object)
        tdnat = np.timedelta64('NaT', 'ns')
        expected = Index([tdnat] * 2, dtype=object)
        if box_with_array is not Index:
            expected = tm.box_expected(expected, box_with_array).astype(object)
            if box_with_array in [Series, DataFrame]:
                msg = "The 'downcast' keyword in fillna is deprecated"
                with tm.assert_produces_warning(FutureWarning, match=msg):
                    expected = expected.fillna(tdnat, downcast=False)
        result = left / right
        tm.assert_equal(result, expected)
        result = left // right
        tm.assert_equal(result, expected)