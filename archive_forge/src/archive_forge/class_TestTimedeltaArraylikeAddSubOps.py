from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
class TestTimedeltaArraylikeAddSubOps:

    def test_sub_nat_retain_unit(self):
        ser = pd.to_timedelta(Series(['00:00:01'])).astype('m8[s]')
        result = ser - NaT
        expected = Series([NaT], dtype='m8[s]')
        tm.assert_series_equal(result, expected)

    def test_timedelta_ops_with_missing_values(self):
        s1 = pd.to_timedelta(Series(['00:00:01']))
        s2 = pd.to_timedelta(Series(['00:00:02']))
        sn = pd.to_timedelta(Series([NaT], dtype='m8[ns]'))
        df1 = DataFrame(['00:00:01']).apply(pd.to_timedelta)
        df2 = DataFrame(['00:00:02']).apply(pd.to_timedelta)
        dfn = DataFrame([NaT._value]).apply(pd.to_timedelta)
        scalar1 = pd.to_timedelta('00:00:01')
        scalar2 = pd.to_timedelta('00:00:02')
        timedelta_NaT = pd.to_timedelta('NaT')
        actual = scalar1 + scalar1
        assert actual == scalar2
        actual = scalar2 - scalar1
        assert actual == scalar1
        actual = s1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - s1
        tm.assert_series_equal(actual, s1)
        actual = s1 + scalar1
        tm.assert_series_equal(actual, s2)
        actual = scalar1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - scalar1
        tm.assert_series_equal(actual, s1)
        actual = -scalar1 + s2
        tm.assert_series_equal(actual, s1)
        actual = s1 + timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)
        actual = s1 - timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = -timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)
        msg = 'unsupported operand type'
        with pytest.raises(TypeError, match=msg):
            s1 + np.nan
        with pytest.raises(TypeError, match=msg):
            np.nan + s1
        with pytest.raises(TypeError, match=msg):
            s1 - np.nan
        with pytest.raises(TypeError, match=msg):
            -np.nan + s1
        actual = s1 + NaT
        tm.assert_series_equal(actual, sn)
        actual = s2 - NaT
        tm.assert_series_equal(actual, sn)
        actual = s1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = s2 - df1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + s1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - s1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - df1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + scalar1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - scalar1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + timedelta_NaT
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - timedelta_NaT
        tm.assert_frame_equal(actual, dfn)
        msg = 'cannot subtract a datelike from|unsupported operand type'
        with pytest.raises(TypeError, match=msg):
            df1 + np.nan
        with pytest.raises(TypeError, match=msg):
            df1 - np.nan
        actual = df1 + NaT
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - NaT
        tm.assert_frame_equal(actual, dfn)

    def test_operators_timedelta64(self):
        v1 = pd.date_range('2012-1-1', periods=3, freq='D')
        v2 = pd.date_range('2012-1-2', periods=3, freq='D')
        rs = Series(v2) - Series(v1)
        xp = Series(1000000000.0 * 3600 * 24, rs.index).astype('int64').astype('timedelta64[ns]')
        tm.assert_series_equal(rs, xp)
        assert rs.dtype == 'timedelta64[ns]'
        df = DataFrame({'A': v1})
        td = Series([timedelta(days=i) for i in range(3)])
        assert td.dtype == 'timedelta64[ns]'
        result = df['A'] - df['A'].shift()
        assert result.dtype == 'timedelta64[ns]'
        result = df['A'] + td
        assert result.dtype == 'M8[ns]'
        maxa = df['A'].max()
        assert isinstance(maxa, Timestamp)
        resultb = df['A'] - df['A'].max()
        assert resultb.dtype == 'timedelta64[ns]'
        result = resultb + df['A']
        values = [Timestamp('20111230'), Timestamp('20120101'), Timestamp('20120103')]
        expected = Series(values, dtype='M8[ns]', name='A')
        tm.assert_series_equal(result, expected)
        result = df['A'] - datetime(2001, 1, 1)
        expected = Series([timedelta(days=4017 + i) for i in range(3)], name='A')
        tm.assert_series_equal(result, expected)
        assert result.dtype == 'm8[ns]'
        d = datetime(2001, 1, 1, 3, 4)
        resulta = df['A'] - d
        assert resulta.dtype == 'm8[ns]'
        resultb = resulta + d
        tm.assert_series_equal(df['A'], resultb)
        td = timedelta(days=1)
        resulta = df['A'] + td
        resultb = resulta - td
        tm.assert_series_equal(resultb, df['A'])
        assert resultb.dtype == 'M8[ns]'
        td = timedelta(minutes=5, seconds=3)
        resulta = df['A'] + td
        resultb = resulta - td
        tm.assert_series_equal(df['A'], resultb)
        assert resultb.dtype == 'M8[ns]'
        value = rs[2] + np.timedelta64(timedelta(minutes=5, seconds=1))
        rs[2] += np.timedelta64(timedelta(minutes=5, seconds=1))
        assert rs[2] == value

    def test_timedelta64_ops_nat(self):
        timedelta_series = Series([NaT, Timedelta('1s')])
        nat_series_dtype_timedelta = Series([NaT, NaT], dtype='timedelta64[ns]')
        single_nat_dtype_timedelta = Series([NaT], dtype='timedelta64[ns]')
        tm.assert_series_equal(timedelta_series - NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(-NaT + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series - single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(-single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta * 1.0, nat_series_dtype_timedelta)
        tm.assert_series_equal(1.0 * nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series * 1, timedelta_series)
        tm.assert_series_equal(1 * timedelta_series, timedelta_series)
        tm.assert_series_equal(timedelta_series * 1.5, Series([NaT, Timedelta('1.5s')]))
        tm.assert_series_equal(1.5 * timedelta_series, Series([NaT, Timedelta('1.5s')]))
        tm.assert_series_equal(timedelta_series * np.nan, nat_series_dtype_timedelta)
        tm.assert_series_equal(np.nan * timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series / 2, Series([NaT, Timedelta('0.5s')]))
        tm.assert_series_equal(timedelta_series / 2.0, Series([NaT, Timedelta('0.5s')]))
        tm.assert_series_equal(timedelta_series / np.nan, nat_series_dtype_timedelta)

    @pytest.mark.parametrize('cls', [Timestamp, datetime, np.datetime64])
    def test_td64arr_add_sub_datetimelike_scalar(self, cls, box_with_array, tz_naive_fixture):
        tz = tz_naive_fixture
        dt_scalar = Timestamp('2012-01-01', tz=tz)
        if cls is datetime:
            ts = dt_scalar.to_pydatetime()
        elif cls is np.datetime64:
            if tz_naive_fixture is not None:
                pytest.skip(f'{cls} doesn support {tz_naive_fixture}')
            ts = dt_scalar.to_datetime64()
        else:
            ts = dt_scalar
        tdi = timedelta_range('1 day', periods=3)
        expected = pd.date_range('2012-01-02', periods=3, tz=tz)
        tdarr = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(ts + tdarr, expected)
        tm.assert_equal(tdarr + ts, expected)
        expected2 = pd.date_range('2011-12-31', periods=3, freq='-1D', tz=tz)
        expected2 = tm.box_expected(expected2, box_with_array)
        tm.assert_equal(ts - tdarr, expected2)
        tm.assert_equal(ts + -tdarr, expected2)
        msg = 'cannot subtract a datelike'
        with pytest.raises(TypeError, match=msg):
            tdarr - ts

    def test_td64arr_add_datetime64_nat(self, box_with_array):
        other = np.datetime64('NaT')
        tdi = timedelta_range('1 day', periods=3)
        expected = DatetimeIndex(['NaT', 'NaT', 'NaT'], dtype='M8[ns]')
        tdser = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(tdser + other, expected)
        tm.assert_equal(other + tdser, expected)

    def test_td64arr_sub_dt64_array(self, box_with_array):
        dti = pd.date_range('2016-01-01', periods=3)
        tdi = TimedeltaIndex(['-1 Day'] * 3)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) - tdi
        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        msg = 'cannot subtract a datelike from'
        with pytest.raises(TypeError, match=msg):
            tdi - dtarr
        result = dtarr - tdi
        tm.assert_equal(result, expected)

    def test_td64arr_add_dt64_array(self, box_with_array):
        dti = pd.date_range('2016-01-01', periods=3)
        tdi = TimedeltaIndex(['-1 Day'] * 3)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) + tdi
        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = tdi + dtarr
        tm.assert_equal(result, expected)
        result = dtarr + tdi
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
    @pytest.mark.parametrize('tdi_freq', [None, 'h'])
    def test_td64arr_sub_periodlike(self, box_with_array, box_with_array2, tdi_freq, pi_freq):
        tdi = TimedeltaIndex(['1 hours', '2 hours'], freq=tdi_freq)
        dti = Timestamp('2018-03-07 17:16:40') + tdi
        pi = dti.to_period(pi_freq)
        per = pi[0]
        tdi = tm.box_expected(tdi, box_with_array)
        pi = tm.box_expected(pi, box_with_array2)
        msg = 'cannot subtract|unsupported operand type'
        with pytest.raises(TypeError, match=msg):
            tdi - pi
        with pytest.raises(TypeError, match=msg):
            tdi - per

    @pytest.mark.parametrize('other', ['a', 1, 1.5, np.array(2)])
    def test_td64arr_addsub_numeric_scalar_invalid(self, box_with_array, other):
        tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        tdarr = tm.box_expected(tdser, box_with_array)
        assert_invalid_addsub_type(tdarr, other)

    @pytest.mark.parametrize('vec', [np.array([1, 2, 3]), Index([1, 2, 3]), Series([1, 2, 3]), DataFrame([[1, 2, 3]])], ids=lambda x: type(x).__name__)
    def test_td64arr_addsub_numeric_arr_invalid(self, box_with_array, vec, any_real_numpy_dtype):
        tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        tdarr = tm.box_expected(tdser, box_with_array)
        vector = vec.astype(any_real_numpy_dtype)
        assert_invalid_addsub_type(tdarr, vector)

    def test_td64arr_add_sub_int(self, box_with_array, one):
        rng = timedelta_range('1 days 09:00:00', freq='h', periods=10)
        tdarr = tm.box_expected(rng, box_with_array)
        msg = 'Addition/subtraction of integers'
        assert_invalid_addsub_type(tdarr, one, msg)
        with pytest.raises(TypeError, match=msg):
            tdarr += one
        with pytest.raises(TypeError, match=msg):
            tdarr -= one

    def test_td64arr_add_sub_integer_array(self, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        rng = timedelta_range('1 days 09:00:00', freq='h', periods=3)
        tdarr = tm.box_expected(rng, box)
        other = tm.box_expected([4, 3, 2], xbox)
        msg = 'Addition/subtraction of integers and integer-arrays'
        assert_invalid_addsub_type(tdarr, other, msg)

    def test_td64arr_addsub_integer_array_no_freq(self, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        tdi = TimedeltaIndex(['1 Day', 'NaT', '3 Hours'])
        tdarr = tm.box_expected(tdi, box)
        other = tm.box_expected([14, -1, 16], xbox)
        msg = 'Addition/subtraction of integers'
        assert_invalid_addsub_type(tdarr, other, msg)

    def test_td64arr_add_sub_td64_array(self, box_with_array):
        box = box_with_array
        dti = pd.date_range('2016-01-01', periods=3)
        tdi = dti - dti.shift(1)
        tdarr = tdi.values
        expected = 2 * tdi
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)
        result = tdi + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + tdi
        tm.assert_equal(result, expected)
        expected_sub = 0 * tdi
        result = tdi - tdarr
        tm.assert_equal(result, expected_sub)
        result = tdarr - tdi
        tm.assert_equal(result, expected_sub)

    def test_td64arr_add_sub_tdi(self, box_with_array, names):
        box = box_with_array
        exname = get_expected_name(box, names)
        tdi = TimedeltaIndex(['0 days', '1 day'], name=names[1])
        tdi = np.array(tdi) if box in [tm.to_array, pd.array] else tdi
        ser = Series([Timedelta(hours=3), Timedelta(hours=4)], name=names[0])
        expected = Series([Timedelta(hours=3), Timedelta(days=1, hours=4)], name=exname)
        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)
        result = tdi + ser
        tm.assert_equal(result, expected)
        assert_dtype(result, 'timedelta64[ns]')
        result = ser + tdi
        tm.assert_equal(result, expected)
        assert_dtype(result, 'timedelta64[ns]')
        expected = Series([Timedelta(hours=-3), Timedelta(days=1, hours=-4)], name=exname)
        expected = tm.box_expected(expected, box)
        result = tdi - ser
        tm.assert_equal(result, expected)
        assert_dtype(result, 'timedelta64[ns]')
        result = ser - tdi
        tm.assert_equal(result, -expected)
        assert_dtype(result, 'timedelta64[ns]')

    @pytest.mark.parametrize('tdnat', [np.timedelta64('NaT'), NaT])
    def test_td64arr_add_sub_td64_nat(self, box_with_array, tdnat):
        box = box_with_array
        tdi = TimedeltaIndex([NaT, Timedelta('1s')])
        expected = TimedeltaIndex(['NaT'] * 2)
        obj = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)
        result = obj + tdnat
        tm.assert_equal(result, expected)
        result = tdnat + obj
        tm.assert_equal(result, expected)
        result = obj - tdnat
        tm.assert_equal(result, expected)
        result = tdnat - obj
        tm.assert_equal(result, expected)

    def test_td64arr_add_timedeltalike(self, two_hours, box_with_array):
        box = box_with_array
        rng = timedelta_range('1 days', '10 days')
        expected = timedelta_range('1 days 02:00:00', '10 days 02:00:00', freq='D')
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)
        result = rng + two_hours
        tm.assert_equal(result, expected)
        result = two_hours + rng
        tm.assert_equal(result, expected)

    def test_td64arr_sub_timedeltalike(self, two_hours, box_with_array):
        box = box_with_array
        rng = timedelta_range('1 days', '10 days')
        expected = timedelta_range('0 days 22:00:00', '9 days 22:00:00')
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)
        result = rng - two_hours
        tm.assert_equal(result, expected)
        result = two_hours - rng
        tm.assert_equal(result, -expected)

    def test_td64arr_add_sub_offset_index(self, names, box_with_array):
        box = box_with_array
        exname = get_expected_name(box, names)
        tdi = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'], name=names[0])
        other = Index([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        other = np.array(other) if box in [tm.to_array, pd.array] else other
        expected = TimedeltaIndex([tdi[n] + other[n] for n in range(len(tdi))], freq='infer', name=exname)
        expected_sub = TimedeltaIndex([tdi[n] - other[n] for n in range(len(tdi))], freq='infer', name=exname)
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box).astype(object, copy=False)
        expected_sub = tm.box_expected(expected_sub, box).astype(object, copy=False)
        with tm.assert_produces_warning(PerformanceWarning):
            res = tdi + other
        tm.assert_equal(res, expected)
        with tm.assert_produces_warning(PerformanceWarning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)
        with tm.assert_produces_warning(PerformanceWarning):
            res_sub = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_add_sub_offset_array(self, box_with_array):
        box = box_with_array
        tdi = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'])
        other = np.array([offsets.Hour(n=1), offsets.Minute(n=-2)])
        expected = TimedeltaIndex([tdi[n] + other[n] for n in range(len(tdi))], freq='infer')
        expected_sub = TimedeltaIndex([tdi[n] - other[n] for n in range(len(tdi))], freq='infer')
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box).astype(object)
        with tm.assert_produces_warning(PerformanceWarning):
            res = tdi + other
        tm.assert_equal(res, expected)
        with tm.assert_produces_warning(PerformanceWarning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)
        expected_sub = tm.box_expected(expected_sub, box_with_array).astype(object)
        with tm.assert_produces_warning(PerformanceWarning):
            res_sub = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_with_offset_series(self, names, box_with_array):
        box = box_with_array
        box2 = Series if box in [Index, tm.to_array, pd.array] else box
        exname = get_expected_name(box, names)
        tdi = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'], name=names[0])
        other = Series([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        expected_add = Series([tdi[n] + other[n] for n in range(len(tdi))], name=exname, dtype=object)
        obj = tm.box_expected(tdi, box)
        expected_add = tm.box_expected(expected_add, box2).astype(object)
        with tm.assert_produces_warning(PerformanceWarning):
            res = obj + other
        tm.assert_equal(res, expected_add)
        with tm.assert_produces_warning(PerformanceWarning):
            res2 = other + obj
        tm.assert_equal(res2, expected_add)
        expected_sub = Series([tdi[n] - other[n] for n in range(len(tdi))], name=exname, dtype=object)
        expected_sub = tm.box_expected(expected_sub, box2).astype(object)
        with tm.assert_produces_warning(PerformanceWarning):
            res3 = obj - other
        tm.assert_equal(res3, expected_sub)

    @pytest.mark.parametrize('obox', [np.array, Index, Series])
    def test_td64arr_addsub_anchored_offset_arraylike(self, obox, box_with_array):
        tdi = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'])
        tdi = tm.box_expected(tdi, box_with_array)
        anchored = obox([offsets.MonthEnd(), offsets.Day(n=2)])
        msg = 'has incorrect type|cannot add the type MonthEnd'
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                tdi + anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                anchored + tdi
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                tdi - anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                anchored - tdi

    def test_td64arr_add_sub_object_array(self, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        tdi = timedelta_range('1 day', periods=3, freq='D')
        tdarr = tm.box_expected(tdi, box)
        other = np.array([Timedelta(days=1), offsets.Day(2), Timestamp('2000-01-04')])
        with tm.assert_produces_warning(PerformanceWarning):
            result = tdarr + other
        expected = Index([Timedelta(days=2), Timedelta(days=4), Timestamp('2000-01-07')])
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)
        msg = 'unsupported operand type|cannot subtract a datelike'
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                tdarr - other
        with tm.assert_produces_warning(PerformanceWarning):
            result = other - tdarr
        expected = Index([Timedelta(0), Timedelta(0), Timestamp('2000-01-01')])
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)