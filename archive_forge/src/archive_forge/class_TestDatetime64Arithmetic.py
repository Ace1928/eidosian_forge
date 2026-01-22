from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
class TestDatetime64Arithmetic:

    @pytest.mark.arm_slow
    def test_dt64arr_add_timedeltalike_scalar(self, tz_naive_fixture, two_hours, box_with_array):
        tz = tz_naive_fixture
        rng = date_range('2000-01-01', '2000-02-01', tz=tz)
        expected = date_range('2000-01-01 02:00', '2000-02-01 02:00', tz=tz)
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = rng + two_hours
        tm.assert_equal(result, expected)
        result = two_hours + rng
        tm.assert_equal(result, expected)
        rng += two_hours
        tm.assert_equal(rng, expected)

    def test_dt64arr_sub_timedeltalike_scalar(self, tz_naive_fixture, two_hours, box_with_array):
        tz = tz_naive_fixture
        rng = date_range('2000-01-01', '2000-02-01', tz=tz)
        expected = date_range('1999-12-31 22:00', '2000-01-31 22:00', tz=tz)
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = rng - two_hours
        tm.assert_equal(result, expected)
        rng -= two_hours
        tm.assert_equal(rng, expected)

    def test_dt64_array_sub_dt_with_different_timezone(self, box_with_array):
        t1 = date_range('20130101', periods=3).tz_localize('US/Eastern')
        t1 = tm.box_expected(t1, box_with_array)
        t2 = Timestamp('20130101').tz_localize('CET')
        tnaive = Timestamp(20130101)
        result = t1 - t2
        expected = TimedeltaIndex(['0 days 06:00:00', '1 days 06:00:00', '2 days 06:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = t2 - t1
        expected = TimedeltaIndex(['-1 days +18:00:00', '-2 days +18:00:00', '-3 days +18:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive
        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64_array_sub_dt64_array_with_different_timezone(self, box_with_array):
        t1 = date_range('20130101', periods=3).tz_localize('US/Eastern')
        t1 = tm.box_expected(t1, box_with_array)
        t2 = date_range('20130101', periods=3).tz_localize('CET')
        t2 = tm.box_expected(t2, box_with_array)
        tnaive = date_range('20130101', periods=3)
        result = t1 - t2
        expected = TimedeltaIndex(['0 days 06:00:00', '0 days 06:00:00', '0 days 06:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = t2 - t1
        expected = TimedeltaIndex(['-1 days +18:00:00', '-1 days +18:00:00', '-1 days +18:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive
        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64arr_add_sub_td64_nat(self, box_with_array, tz_naive_fixture):
        tz = tz_naive_fixture
        dti = date_range('1994-04-01', periods=9, tz=tz, freq='QS')
        other = np.timedelta64('NaT')
        expected = DatetimeIndex(['NaT'] * 9, tz=tz).as_unit('ns')
        obj = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = obj + other
        tm.assert_equal(result, expected)
        result = other + obj
        tm.assert_equal(result, expected)
        result = obj - other
        tm.assert_equal(result, expected)
        msg = 'cannot subtract'
        with pytest.raises(TypeError, match=msg):
            other - obj

    def test_dt64arr_add_sub_td64ndarray(self, tz_naive_fixture, box_with_array):
        tz = tz_naive_fixture
        dti = date_range('2016-01-01', periods=3, tz=tz)
        tdi = TimedeltaIndex(['-1 Day', '-1 Day', '-1 Day'])
        tdarr = tdi.values
        expected = date_range('2015-12-31', '2016-01-02', periods=3, tz=tz)
        dtarr = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = dtarr + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + dtarr
        tm.assert_equal(result, expected)
        expected = date_range('2016-01-02', '2016-01-04', periods=3, tz=tz)
        expected = tm.box_expected(expected, box_with_array)
        result = dtarr - tdarr
        tm.assert_equal(result, expected)
        msg = 'cannot subtract|(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            tdarr - dtarr

    @pytest.mark.parametrize('ts', [Timestamp('2013-01-01'), Timestamp('2013-01-01').to_pydatetime(), Timestamp('2013-01-01').to_datetime64(), np.datetime64('2013-01-01', 'D')])
    def test_dt64arr_sub_dtscalar(self, box_with_array, ts):
        idx = date_range('2013-01-01', periods=3)._with_freq(None)
        idx = tm.box_expected(idx, box_with_array)
        expected = TimedeltaIndex(['0 Days', '1 Day', '2 Days'])
        expected = tm.box_expected(expected, box_with_array)
        result = idx - ts
        tm.assert_equal(result, expected)
        result = ts - idx
        tm.assert_equal(result, -expected)
        tm.assert_equal(result, -expected)

    def test_dt64arr_sub_timestamp_tzaware(self, box_with_array):
        ser = date_range('2014-03-17', periods=2, freq='D', tz='US/Eastern')
        ser = ser._with_freq(None)
        ts = ser[0]
        ser = tm.box_expected(ser, box_with_array)
        delta_series = Series([np.timedelta64(0, 'D'), np.timedelta64(1, 'D')])
        expected = tm.box_expected(delta_series, box_with_array)
        tm.assert_equal(ser - ts, expected)
        tm.assert_equal(ts - ser, -expected)

    def test_dt64arr_sub_NaT(self, box_with_array, unit):
        dti = DatetimeIndex([NaT, Timestamp('19900315')]).as_unit(unit)
        ser = tm.box_expected(dti, box_with_array)
        result = ser - NaT
        expected = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        dti_tz = dti.tz_localize('Asia/Tokyo')
        ser_tz = tm.box_expected(dti_tz, box_with_array)
        result = ser_tz - NaT
        expected = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dt64arr_sub_dt64object_array(self, box_with_array, tz_naive_fixture):
        dti = date_range('2016-01-01', periods=3, tz=tz_naive_fixture)
        expected = dti - dti
        obj = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array).astype(object)
        with tm.assert_produces_warning(PerformanceWarning):
            result = obj - obj.astype(object)
        tm.assert_equal(result, expected)

    def test_dt64arr_naive_sub_dt64ndarray(self, box_with_array):
        dti = date_range('2016-01-01', periods=3, tz=None)
        dt64vals = dti.values
        dtarr = tm.box_expected(dti, box_with_array)
        expected = dtarr - dtarr
        result = dtarr - dt64vals
        tm.assert_equal(result, expected)
        result = dt64vals - dtarr
        tm.assert_equal(result, expected)

    def test_dt64arr_aware_sub_dt64ndarray_raises(self, tz_aware_fixture, box_with_array):
        tz = tz_aware_fixture
        dti = date_range('2016-01-01', periods=3, tz=tz)
        dt64vals = dti.values
        dtarr = tm.box_expected(dti, box_with_array)
        msg = 'Cannot subtract tz-naive and tz-aware datetime'
        with pytest.raises(TypeError, match=msg):
            dtarr - dt64vals
        with pytest.raises(TypeError, match=msg):
            dt64vals - dtarr

    def test_dt64arr_add_dtlike_raises(self, tz_naive_fixture, box_with_array):
        tz = tz_naive_fixture
        dti = date_range('2016-01-01', periods=3, tz=tz)
        if tz is None:
            dti2 = dti.tz_localize('US/Eastern')
        else:
            dti2 = dti.tz_localize(None)
        dtarr = tm.box_expected(dti, box_with_array)
        assert_cannot_add(dtarr, dti.values)
        assert_cannot_add(dtarr, dti)
        assert_cannot_add(dtarr, dtarr)
        assert_cannot_add(dtarr, dti[0])
        assert_cannot_add(dtarr, dti[0].to_pydatetime())
        assert_cannot_add(dtarr, dti[0].to_datetime64())
        assert_cannot_add(dtarr, dti2[0])
        assert_cannot_add(dtarr, dti2[0].to_pydatetime())
        assert_cannot_add(dtarr, np.datetime64('2011-01-01', 'D'))

    @pytest.mark.parametrize('freq', ['h', 'D', 'W', '2ME', 'MS', 'QE', 'B', None])
    @pytest.mark.parametrize('dtype', [None, 'uint8'])
    def test_dt64arr_addsub_intlike(self, request, dtype, index_or_series_or_array, freq, tz_naive_fixture):
        tz = tz_naive_fixture
        if freq is None:
            dti = DatetimeIndex(['NaT', '2017-04-05 06:07:08'], tz=tz)
        else:
            dti = date_range('2016-01-01', periods=2, freq=freq, tz=tz)
        obj = index_or_series_or_array(dti)
        other = np.array([4, -1])
        if dtype is not None:
            other = other.astype(dtype)
        msg = '|'.join(['Addition/subtraction of integers', 'cannot subtract DatetimeArray from', 'can only perform ops with numeric values', 'unsupported operand type.*Categorical', "unsupported operand type\\(s\\) for -: 'int' and 'Timestamp'"])
        assert_invalid_addsub_type(obj, 1, msg)
        assert_invalid_addsub_type(obj, np.int64(2), msg)
        assert_invalid_addsub_type(obj, np.array(3, dtype=np.int64), msg)
        assert_invalid_addsub_type(obj, other, msg)
        assert_invalid_addsub_type(obj, np.array(other), msg)
        assert_invalid_addsub_type(obj, pd.array(other), msg)
        assert_invalid_addsub_type(obj, pd.Categorical(other), msg)
        assert_invalid_addsub_type(obj, pd.Index(other), msg)
        assert_invalid_addsub_type(obj, Series(other), msg)

    @pytest.mark.parametrize('other', [3.14, np.array([2.0, 3.0]), Period('2011-01-01', freq='D'), time(1, 2, 3)])
    @pytest.mark.parametrize('dti_freq', [None, 'D'])
    def test_dt64arr_add_sub_invalid(self, dti_freq, other, box_with_array):
        dti = DatetimeIndex(['2011-01-01', '2011-01-02'], freq=dti_freq)
        dtarr = tm.box_expected(dti, box_with_array)
        msg = '|'.join(['unsupported operand type', 'cannot (add|subtract)', 'cannot use operands with types', "ufunc '?(add|subtract)'? cannot use operands with types", 'Concatenation operation is not implemented for NumPy arrays'])
        assert_invalid_addsub_type(dtarr, other, msg)

    @pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
    @pytest.mark.parametrize('dti_freq', [None, 'D'])
    def test_dt64arr_add_sub_parr(self, dti_freq, pi_freq, box_with_array, box_with_array2):
        dti = DatetimeIndex(['2011-01-01', '2011-01-02'], freq=dti_freq)
        pi = dti.to_period(pi_freq)
        dtarr = tm.box_expected(dti, box_with_array)
        parr = tm.box_expected(pi, box_with_array2)
        msg = '|'.join(['cannot (add|subtract)', 'unsupported operand', 'descriptor.*requires', 'ufunc.*cannot use operands'])
        assert_invalid_addsub_type(dtarr, parr, msg)

    @pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
    def test_dt64arr_addsub_time_objects_raises(self, box_with_array, tz_naive_fixture):
        tz = tz_naive_fixture
        obj1 = date_range('2012-01-01', periods=3, tz=tz)
        obj2 = [time(i, i, i) for i in range(3)]
        obj1 = tm.box_expected(obj1, box_with_array)
        obj2 = tm.box_expected(obj2, box_with_array)
        msg = '|'.join(['unsupported operand', 'cannot subtract DatetimeArray from ndarray'])
        assert_invalid_addsub_type(obj1, obj2, msg=msg)

    @pytest.mark.parametrize('dt64_series', [Series([Timestamp('19900315'), Timestamp('19900315')]), Series([NaT, Timestamp('19900315')]), Series([NaT, NaT], dtype='datetime64[ns]')])
    @pytest.mark.parametrize('one', [1, 1.0, np.array(1)])
    def test_dt64_mul_div_numeric_invalid(self, one, dt64_series, box_with_array):
        obj = tm.box_expected(dt64_series, box_with_array)
        msg = 'cannot perform .* with this index type'
        with pytest.raises(TypeError, match=msg):
            obj * one
        with pytest.raises(TypeError, match=msg):
            one * obj
        with pytest.raises(TypeError, match=msg):
            obj / one
        with pytest.raises(TypeError, match=msg):
            one / obj