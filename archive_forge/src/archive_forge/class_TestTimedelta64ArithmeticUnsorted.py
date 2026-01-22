from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
class TestTimedelta64ArithmeticUnsorted:

    def test_ufunc_coercions(self):
        idx = TimedeltaIndex(['2h', '4h', '6h', '8h', '10h'], freq='2h', name='x')
        for result in [idx * 2, np.multiply(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(['4h', '8h', '12h', '16h', '20h'], freq='4h', name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq == '4h'
        for result in [idx / 2, np.divide(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(['1h', '2h', '3h', '4h', '5h'], freq='h', name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq == 'h'
        for result in [-idx, np.negative(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(['-2h', '-4h', '-6h', '-8h', '-10h'], freq='-2h', name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq == '-2h'
        idx = TimedeltaIndex(['-2h', '-1h', '0h', '1h', '2h'], freq='h', name='x')
        for result in [abs(idx), np.absolute(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(['2h', '1h', '0h', '1h', '2h'], freq=None, name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq is None

    def test_subtraction_ops(self):
        tdi = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        dti = pd.date_range('20130101', periods=3, name='bar')
        td = Timedelta('1 days')
        dt = Timestamp('20130101')
        msg = 'cannot subtract a datelike from a TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdi - dt
        with pytest.raises(TypeError, match=msg):
            tdi - dti
        msg = 'unsupported operand type\\(s\\) for -'
        with pytest.raises(TypeError, match=msg):
            td - dt
        msg = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            td - dti
        result = dt - dti
        expected = TimedeltaIndex(['0 days', '-1 days', '-2 days'], name='bar')
        tm.assert_index_equal(result, expected)
        result = dti - dt
        expected = TimedeltaIndex(['0 days', '1 days', '2 days'], name='bar')
        tm.assert_index_equal(result, expected)
        result = tdi - td
        expected = TimedeltaIndex(['0 days', NaT, '1 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result = td - tdi
        expected = TimedeltaIndex(['0 days', NaT, '-1 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result = dti - td
        expected = DatetimeIndex(['20121231', '20130101', '20130102'], dtype='M8[ns]', freq='D', name='bar')
        tm.assert_index_equal(result, expected)
        result = dt - tdi
        expected = DatetimeIndex(['20121231', NaT, '20121230'], dtype='M8[ns]', name='foo')
        tm.assert_index_equal(result, expected)

    def test_subtraction_ops_with_tz(self, box_with_array):
        dti = pd.date_range('20130101', periods=3)
        dti = tm.box_expected(dti, box_with_array)
        ts = Timestamp('20130101')
        dt = ts.to_pydatetime()
        dti_tz = pd.date_range('20130101', periods=3).tz_localize('US/Eastern')
        dti_tz = tm.box_expected(dti_tz, box_with_array)
        ts_tz = Timestamp('20130101').tz_localize('US/Eastern')
        ts_tz2 = Timestamp('20130101').tz_localize('CET')
        dt_tz = ts_tz.to_pydatetime()
        td = Timedelta('1 days')

        def _check(result, expected):
            assert result == expected
            assert isinstance(result, Timedelta)
        result = ts - ts
        expected = Timedelta('0 days')
        _check(result, expected)
        result = dt_tz - ts_tz
        expected = Timedelta('0 days')
        _check(result, expected)
        result = ts_tz - dt_tz
        expected = Timedelta('0 days')
        _check(result, expected)
        msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects.'
        with pytest.raises(TypeError, match=msg):
            dt_tz - ts
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt_tz - dt
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt - dt_tz
        msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects.'
        with pytest.raises(TypeError, match=msg):
            ts - dt_tz
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - ts
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - dt
        msg = 'Cannot subtract tz-naive and tz-aware'
        with pytest.raises(TypeError, match=msg):
            dti - ts_tz
        with pytest.raises(TypeError, match=msg):
            dti_tz - ts
        result = dti_tz - dt_tz
        expected = TimedeltaIndex(['0 days', '1 days', '2 days'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = dt_tz - dti_tz
        expected = TimedeltaIndex(['0 days', '-1 days', '-2 days'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = dti_tz - ts_tz
        expected = TimedeltaIndex(['0 days', '1 days', '2 days'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = ts_tz - dti_tz
        expected = TimedeltaIndex(['0 days', '-1 days', '-2 days'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = td - td
        expected = Timedelta('0 days')
        _check(result, expected)
        result = dti_tz - td
        expected = DatetimeIndex(['20121231', '20130101', '20130102'], tz='US/Eastern').as_unit('ns')
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dti_tdi_numeric_ops(self):
        tdi = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        dti = pd.date_range('20130101', periods=3, name='bar')
        result = tdi - tdi
        expected = TimedeltaIndex(['0 days', NaT, '0 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result = tdi + tdi
        expected = TimedeltaIndex(['2 days', NaT, '4 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result = dti - tdi
        expected = DatetimeIndex(['20121231', NaT, '20130101'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)

    def test_addition_ops(self):
        tdi = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        dti = pd.date_range('20130101', periods=3, name='bar')
        td = Timedelta('1 days')
        dt = Timestamp('20130101')
        result = tdi + dt
        expected = DatetimeIndex(['20130102', NaT, '20130103'], dtype='M8[ns]', name='foo')
        tm.assert_index_equal(result, expected)
        result = dt + tdi
        expected = DatetimeIndex(['20130102', NaT, '20130103'], dtype='M8[ns]', name='foo')
        tm.assert_index_equal(result, expected)
        result = td + tdi
        expected = TimedeltaIndex(['2 days', NaT, '3 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result = tdi + td
        expected = TimedeltaIndex(['2 days', NaT, '3 days'], name='foo')
        tm.assert_index_equal(result, expected)
        msg = 'cannot add indices of unequal length'
        with pytest.raises(ValueError, match=msg):
            tdi + dti[0:1]
        with pytest.raises(ValueError, match=msg):
            tdi[0:1] + dti
        msg = 'Addition/subtraction of integers and integer-arrays'
        with pytest.raises(TypeError, match=msg):
            tdi + Index([1, 2, 3], dtype=np.int64)
        result = tdi + dti
        expected = DatetimeIndex(['20130102', NaT, '20130105'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)
        result = dti + tdi
        expected = DatetimeIndex(['20130102', NaT, '20130105'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)
        result = dt + td
        expected = Timestamp('20130102')
        assert result == expected
        result = td + dt
        expected = Timestamp('20130102')
        assert result == expected

    @pytest.mark.parametrize('freq', ['D', 'B'])
    def test_timedelta(self, freq):
        index = pd.date_range('1/1/2000', periods=50, freq=freq)
        shifted = index + timedelta(1)
        back = shifted + timedelta(-1)
        back = back._with_freq('infer')
        tm.assert_index_equal(index, back)
        if freq == 'D':
            expected = pd.tseries.offsets.Day(1)
            assert index.freq == expected
            assert shifted.freq == expected
            assert back.freq == expected
        else:
            assert index.freq == pd.tseries.offsets.BusinessDay(1)
            assert shifted.freq is None
            assert back.freq == pd.tseries.offsets.BusinessDay(1)
        result = index - timedelta(1)
        expected = index + timedelta(-1)
        tm.assert_index_equal(result, expected)

    def test_timedelta_tick_arithmetic(self):
        rng = pd.date_range('2013', '2014')
        s = Series(rng)
        result1 = rng - offsets.Hour(1)
        result2 = DatetimeIndex(s - np.timedelta64(100000000))
        result3 = rng - np.timedelta64(100000000)
        result4 = DatetimeIndex(s - offsets.Hour(1))
        assert result1.freq == rng.freq
        result1 = result1._with_freq(None)
        tm.assert_index_equal(result1, result4)
        assert result3.freq == rng.freq
        result3 = result3._with_freq(None)
        tm.assert_index_equal(result2, result3)

    def test_tda_add_sub_index(self):
        tdi = TimedeltaIndex(['1 days', NaT, '2 days'])
        tda = tdi.array
        dti = pd.date_range('1999-12-31', periods=3, freq='D')
        result = tda + dti
        expected = tdi + dti
        tm.assert_index_equal(result, expected)
        result = tda + tdi
        expected = tdi + tdi
        tm.assert_index_equal(result, expected)
        result = tda - tdi
        expected = tdi - tdi
        tm.assert_index_equal(result, expected)

    def test_tda_add_dt64_object_array(self, box_with_array, tz_naive_fixture):
        box = box_with_array
        dti = pd.date_range('2016-01-01', periods=3, tz=tz_naive_fixture)
        dti = dti._with_freq(None)
        tdi = dti - dti
        obj = tm.box_expected(tdi, box)
        other = tm.box_expected(dti, box)
        with tm.assert_produces_warning(PerformanceWarning):
            result = obj + other.astype(object)
        tm.assert_equal(result, other.astype(object))

    def test_tdi_iadd_timedeltalike(self, two_hours, box_with_array):
        rng = timedelta_range('1 days', '10 days')
        expected = timedelta_range('1 days 02:00:00', '10 days 02:00:00', freq='D')
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        orig_rng = rng
        rng += two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            tm.assert_equal(orig_rng, expected)

    def test_tdi_isub_timedeltalike(self, two_hours, box_with_array):
        rng = timedelta_range('1 days', '10 days')
        expected = timedelta_range('0 days 22:00:00', '9 days 22:00:00')
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        orig_rng = rng
        rng -= two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            tm.assert_equal(orig_rng, expected)

    def test_tdi_ops_attributes(self):
        rng = timedelta_range('2 days', periods=5, freq='2D', name='x')
        result = rng + 1 * rng.freq
        exp = timedelta_range('4 days', periods=5, freq='2D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == '2D'
        result = rng - 2 * rng.freq
        exp = timedelta_range('-2 days', periods=5, freq='2D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == '2D'
        result = rng * 2
        exp = timedelta_range('4 days', periods=5, freq='4D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == '4D'
        result = rng / 2
        exp = timedelta_range('1 days', periods=5, freq='D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == 'D'
        result = -rng
        exp = timedelta_range('-2 days', periods=5, freq='-2D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == '-2D'
        rng = timedelta_range('-2 days', periods=5, freq='D', name='x')
        result = abs(rng)
        exp = TimedeltaIndex(['2 days', '1 days', '0 days', '1 days', '2 days'], name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq is None