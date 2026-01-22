from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
class TestDateRanges:

    def test_date_range_name(self):
        idx = date_range(start='2000-01-01', periods=1, freq='YE', name='TEST')
        assert idx.name == 'TEST'

    def test_date_range_invalid_periods(self):
        msg = 'periods must be a number, got foo'
        with pytest.raises(TypeError, match=msg):
            date_range(start='1/1/2000', periods='foo', freq='D')

    def test_date_range_fractional_period(self):
        msg = "Non-integer 'periods' in pd.date_range, pd.timedelta_range"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rng = date_range('1/1/2000', periods=10.5)
        exp = date_range('1/1/2000', periods=10)
        tm.assert_index_equal(rng, exp)

    @pytest.mark.parametrize('freq,freq_depr', [('2ME', '2M'), ('2SME', '2SM'), ('2BQE', '2BQ'), ('2BYE', '2BY')])
    def test_date_range_frequency_M_SM_BQ_BY_deprecated(self, freq, freq_depr):
        depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
        f"in a future version, please use '{freq[1:]}' instead."
        expected = date_range('1/1/2000', periods=4, freq=freq)
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = date_range('1/1/2000', periods=4, freq=freq_depr)
        tm.assert_index_equal(result, expected)

    def test_date_range_tuple_freq_raises(self):
        edate = datetime(2000, 1, 1)
        with pytest.raises(TypeError, match='pass as a string instead'):
            date_range(end=edate, freq=('D', 5), periods=20)

    @pytest.mark.parametrize('freq', ['ns', 'us', 'ms', 'min', 's', 'h', 'D'])
    def test_date_range_edges(self, freq):
        td = Timedelta(f'1{freq}')
        ts = Timestamp('1970-01-01')
        idx = date_range(start=ts + td, end=ts + 4 * td, freq=freq)
        exp = DatetimeIndex([ts + n * td for n in range(1, 5)], dtype='M8[ns]', freq=freq)
        tm.assert_index_equal(idx, exp)
        idx = date_range(start=ts + 4 * td, end=ts + td, freq=freq)
        exp = DatetimeIndex([], dtype='M8[ns]', freq=freq)
        tm.assert_index_equal(idx, exp)
        idx = date_range(start=ts + td, end=ts + td, freq=freq)
        exp = DatetimeIndex([ts + td], dtype='M8[ns]', freq=freq)
        tm.assert_index_equal(idx, exp)

    def test_date_range_near_implementation_bound(self):
        freq = Timedelta(1)
        with pytest.raises(OutOfBoundsDatetime, match='Cannot generate range with'):
            date_range(end=Timestamp.min, periods=2, freq=freq)

    def test_date_range_nat(self):
        msg = 'Neither `start` nor `end` can be NaT'
        with pytest.raises(ValueError, match=msg):
            date_range(start='2016-01-01', end=pd.NaT, freq='D')
        with pytest.raises(ValueError, match=msg):
            date_range(start=pd.NaT, end='2016-01-01', freq='D')

    def test_date_range_multiplication_overflow(self):
        with tm.assert_produces_warning(None):
            dti = date_range(start='1677-09-22', periods=213503, freq='D')
        assert dti[0] == Timestamp('1677-09-22')
        assert len(dti) == 213503
        msg = 'Cannot generate range with'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range('1969-05-04', periods=200000000, freq='30000D')

    def test_date_range_unsigned_overflow_handling(self):
        dti = date_range(start='1677-09-22', end='2262-04-11', freq='D')
        dti2 = date_range(start=dti[0], periods=len(dti), freq='D')
        assert dti2.equals(dti)
        dti3 = date_range(end=dti[-1], periods=len(dti), freq='D')
        assert dti3.equals(dti)

    def test_date_range_int64_overflow_non_recoverable(self):
        msg = 'Cannot generate range with'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(start='1970-02-01', periods=106752 * 24, freq='h')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(end='1969-11-14', periods=106752 * 24, freq='h')

    @pytest.mark.slow
    @pytest.mark.parametrize('s_ts, e_ts', [('2262-02-23', '1969-11-14'), ('1970-02-01', '1677-10-22')])
    def test_date_range_int64_overflow_stride_endpoint_different_signs(self, s_ts, e_ts):
        start = Timestamp(s_ts)
        end = Timestamp(e_ts)
        expected = date_range(start=start, end=end, freq='-1h')
        assert expected[0] == start
        assert expected[-1] == end
        dti = date_range(end=end, periods=len(expected), freq='-1h')
        tm.assert_index_equal(dti, expected)

    def test_date_range_out_of_bounds(self):
        msg = 'Cannot generate range'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range('2016-01-01', periods=100000, freq='D')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(end='1763-10-12', periods=100000, freq='D')

    def test_date_range_gen_error(self):
        rng = date_range('1/1/2000 00:00', '1/1/2000 00:18', freq='5min')
        assert len(rng) == 4

    def test_date_range_normalize(self):
        snap = datetime.today()
        n = 50
        rng = date_range(snap, periods=n, normalize=False, freq='2D')
        offset = timedelta(2)
        expected = DatetimeIndex([snap + i * offset for i in range(n)], dtype='M8[ns]', freq=offset)
        tm.assert_index_equal(rng, expected)
        rng = date_range('1/1/2000 08:15', periods=n, normalize=False, freq='B')
        the_time = time(8, 15)
        for val in rng:
            assert val.time() == the_time

    def test_date_range_ambiguous_arguments(self):
        start = datetime(2011, 1, 1, 5, 3, 40)
        end = datetime(2011, 1, 1, 8, 9, 40)
        msg = 'Of the four parameters: start, end, periods, and freq, exactly three must be specified'
        with pytest.raises(ValueError, match=msg):
            date_range(start, end, periods=10, freq='s')

    def test_date_range_convenience_periods(self, unit):
        result = date_range('2018-04-24', '2018-04-27', periods=3, unit=unit)
        expected = DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00', '2018-04-27 00:00:00'], dtype=f'M8[{unit}]', freq=None)
        tm.assert_index_equal(result, expected)
        result = date_range('2018-04-01 01:00:00', '2018-04-01 04:00:00', tz='Australia/Sydney', periods=3, unit=unit)
        expected = DatetimeIndex([Timestamp('2018-04-01 01:00:00+1100', tz='Australia/Sydney'), Timestamp('2018-04-01 02:00:00+1000', tz='Australia/Sydney'), Timestamp('2018-04-01 04:00:00+1000', tz='Australia/Sydney')]).as_unit(unit)
        tm.assert_index_equal(result, expected)

    def test_date_range_index_comparison(self):
        rng = date_range('2011-01-01', periods=3, tz='US/Eastern')
        df = Series(rng).to_frame()
        arr = np.array([rng.to_list()]).T
        arr2 = np.array([rng]).T
        with pytest.raises(ValueError, match='Unable to coerce to Series'):
            rng == df
        with pytest.raises(ValueError, match='Unable to coerce to Series'):
            df == rng
        expected = DataFrame([True, True, True])
        results = df == arr2
        tm.assert_frame_equal(results, expected)
        expected = Series([True, True, True], name=0)
        results = df[0] == arr2[:, 0]
        tm.assert_series_equal(results, expected)
        expected = np.array([[True, False, False], [False, True, False], [False, False, True]])
        results = rng == arr
        tm.assert_numpy_array_equal(results, expected)

    @pytest.mark.parametrize('start,end,result_tz', [['20180101', '20180103', 'US/Eastern'], [datetime(2018, 1, 1), datetime(2018, 1, 3), 'US/Eastern'], [Timestamp('20180101'), Timestamp('20180103'), 'US/Eastern'], [Timestamp('20180101', tz='US/Eastern'), Timestamp('20180103', tz='US/Eastern'), 'US/Eastern'], [Timestamp('20180101', tz='US/Eastern'), Timestamp('20180103', tz='US/Eastern'), None]])
    def test_date_range_linspacing_tz(self, start, end, result_tz):
        result = date_range(start, end, periods=3, tz=result_tz)
        expected = date_range('20180101', periods=3, freq='D', tz='US/Eastern')
        tm.assert_index_equal(result, expected)

    def test_date_range_timedelta(self):
        start = '2020-01-01'
        end = '2020-01-11'
        rng1 = date_range(start, end, freq='3D')
        rng2 = date_range(start, end, freq=timedelta(days=3))
        tm.assert_index_equal(rng1, rng2)

    def test_range_misspecified(self):
        msg = 'Of the four parameters: start, end, periods, and freq, exactly three must be specified'
        with pytest.raises(ValueError, match=msg):
            date_range(start='1/1/2000')
        with pytest.raises(ValueError, match=msg):
            date_range(end='1/1/2000')
        with pytest.raises(ValueError, match=msg):
            date_range(periods=10)
        with pytest.raises(ValueError, match=msg):
            date_range(start='1/1/2000', freq='h')
        with pytest.raises(ValueError, match=msg):
            date_range(end='1/1/2000', freq='h')
        with pytest.raises(ValueError, match=msg):
            date_range(periods=10, freq='h')
        with pytest.raises(ValueError, match=msg):
            date_range()

    def test_compat_replace(self):
        result = date_range(Timestamp('1960-04-01 00:00:00'), periods=76, freq='QS-JAN')
        assert len(result) == 76

    def test_catch_infinite_loop(self):
        offset = offsets.DateOffset(minute=5)
        msg = 'Offset <DateOffset: minute=5> did not increment date'
        with pytest.raises(ValueError, match=msg):
            date_range(datetime(2011, 11, 11), datetime(2011, 11, 12), freq=offset)

    def test_construct_over_dst(self, unit):
        pre_dst = Timestamp('2010-11-07 01:00:00').tz_localize('US/Pacific', ambiguous=True)
        pst_dst = Timestamp('2010-11-07 01:00:00').tz_localize('US/Pacific', ambiguous=False)
        expect_data = [Timestamp('2010-11-07 00:00:00', tz='US/Pacific'), pre_dst, pst_dst]
        expected = DatetimeIndex(expect_data, freq='h').as_unit(unit)
        result = date_range(start='2010-11-7', periods=3, freq='h', tz='US/Pacific', unit=unit)
        tm.assert_index_equal(result, expected)

    def test_construct_with_different_start_end_string_format(self, unit):
        result = date_range('2013-01-01 00:00:00+09:00', '2013/01/01 02:00:00+09:00', freq='h', unit=unit)
        expected = DatetimeIndex([Timestamp('2013-01-01 00:00:00+09:00'), Timestamp('2013-01-01 01:00:00+09:00'), Timestamp('2013-01-01 02:00:00+09:00')], freq='h').as_unit(unit)
        tm.assert_index_equal(result, expected)

    def test_error_with_zero_monthends(self):
        msg = 'Offset <0 \\* MonthEnds> did not increment date'
        with pytest.raises(ValueError, match=msg):
            date_range('1/1/2000', '1/1/2001', freq=MonthEnd(0))

    def test_range_bug(self, unit):
        offset = DateOffset(months=3)
        result = date_range('2011-1-1', '2012-1-31', freq=offset, unit=unit)
        start = datetime(2011, 1, 1)
        expected = DatetimeIndex([start + i * offset for i in range(5)], dtype=f'M8[{unit}]', freq=offset)
        tm.assert_index_equal(result, expected)

    def test_range_tz_pytz(self):
        tz = timezone('US/Eastern')
        start = tz.localize(datetime(2011, 1, 1))
        end = tz.localize(datetime(2011, 1, 3))
        dr = date_range(start=start, periods=3)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end
        dr = date_range(end=end, periods=3)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end
        dr = date_range(start=start, end=end)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end

    @pytest.mark.parametrize('start, end', [[Timestamp(datetime(2014, 3, 6), tz='US/Eastern'), Timestamp(datetime(2014, 3, 12), tz='US/Eastern')], [Timestamp(datetime(2013, 11, 1), tz='US/Eastern'), Timestamp(datetime(2013, 11, 6), tz='US/Eastern')]])
    def test_range_tz_dst_straddle_pytz(self, start, end):
        dr = date_range(start, end, freq='D')
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)
        dr = date_range(start, end, freq='D', tz='US/Eastern')
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)
        dr = date_range(start.replace(tzinfo=None), end.replace(tzinfo=None), freq='D', tz='US/Eastern')
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)

    def test_range_tz_dateutil(self):
        from pandas._libs.tslibs.timezones import maybe_get_tz
        tz = lambda x: maybe_get_tz('dateutil/' + x)
        start = datetime(2011, 1, 1, tzinfo=tz('US/Eastern'))
        end = datetime(2011, 1, 3, tzinfo=tz('US/Eastern'))
        dr = date_range(start=start, periods=3)
        assert dr.tz == tz('US/Eastern')
        assert dr[0] == start
        assert dr[2] == end
        dr = date_range(end=end, periods=3)
        assert dr.tz == tz('US/Eastern')
        assert dr[0] == start
        assert dr[2] == end
        dr = date_range(start=start, end=end)
        assert dr.tz == tz('US/Eastern')
        assert dr[0] == start
        assert dr[2] == end

    @pytest.mark.parametrize('freq', ['1D', '3D', '2ME', '7W', '3h', 'YE'])
    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_range_closed(self, freq, tz, inclusive_endpoints_fixture):
        begin = Timestamp('2011/1/1', tz=tz)
        end = Timestamp('2014/1/1', tz=tz)
        result_range = date_range(begin, end, inclusive=inclusive_endpoints_fixture, freq=freq)
        both_range = date_range(begin, end, inclusive='both', freq=freq)
        expected_range = _get_expected_range(begin, end, both_range, inclusive_endpoints_fixture)
        tm.assert_index_equal(expected_range, result_range)

    @pytest.mark.parametrize('freq', ['1D', '3D', '2ME', '7W', '3h', 'YE'])
    def test_range_with_tz_closed_with_tz_aware_start_end(self, freq, inclusive_endpoints_fixture):
        begin = Timestamp('2011/1/1')
        end = Timestamp('2014/1/1')
        begintz = Timestamp('2011/1/1', tz='US/Eastern')
        endtz = Timestamp('2014/1/1', tz='US/Eastern')
        result_range = date_range(begin, end, inclusive=inclusive_endpoints_fixture, freq=freq, tz='US/Eastern')
        both_range = date_range(begin, end, inclusive='both', freq=freq, tz='US/Eastern')
        expected_range = _get_expected_range(begintz, endtz, both_range, inclusive_endpoints_fixture)
        tm.assert_index_equal(expected_range, result_range)

    def test_range_closed_boundary(self, inclusive_endpoints_fixture):
        right_boundary = date_range('2015-09-12', '2015-12-01', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
        left_boundary = date_range('2015-09-01', '2015-09-12', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
        both_boundary = date_range('2015-09-01', '2015-12-01', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
        neither_boundary = date_range('2015-09-11', '2015-09-12', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
        expected_right = both_boundary
        expected_left = both_boundary
        expected_both = both_boundary
        if inclusive_endpoints_fixture == 'right':
            expected_left = both_boundary[1:]
        elif inclusive_endpoints_fixture == 'left':
            expected_right = both_boundary[:-1]
        elif inclusive_endpoints_fixture == 'both':
            expected_right = both_boundary[1:]
            expected_left = both_boundary[:-1]
        expected_neither = both_boundary[1:-1]
        tm.assert_index_equal(right_boundary, expected_right)
        tm.assert_index_equal(left_boundary, expected_left)
        tm.assert_index_equal(both_boundary, expected_both)
        tm.assert_index_equal(neither_boundary, expected_neither)

    def test_date_range_years_only(self, tz_naive_fixture):
        tz = tz_naive_fixture
        rng1 = date_range('2014', '2015', freq='ME', tz=tz)
        expected1 = date_range('2014-01-31', '2014-12-31', freq='ME', tz=tz)
        tm.assert_index_equal(rng1, expected1)
        rng2 = date_range('2014', '2015', freq='MS', tz=tz)
        expected2 = date_range('2014-01-01', '2015-01-01', freq='MS', tz=tz)
        tm.assert_index_equal(rng2, expected2)
        rng3 = date_range('2014', '2020', freq='YE', tz=tz)
        expected3 = date_range('2014-12-31', '2019-12-31', freq='YE', tz=tz)
        tm.assert_index_equal(rng3, expected3)
        rng4 = date_range('2014', '2020', freq='YS', tz=tz)
        expected4 = date_range('2014-01-01', '2020-01-01', freq='YS', tz=tz)
        tm.assert_index_equal(rng4, expected4)

    def test_freq_divides_end_in_nanos(self):
        result_1 = date_range('2005-01-12 10:00', '2005-01-12 16:00', freq='345min')
        result_2 = date_range('2005-01-13 10:00', '2005-01-13 16:00', freq='345min')
        expected_1 = DatetimeIndex(['2005-01-12 10:00:00', '2005-01-12 15:45:00'], dtype='datetime64[ns]', freq='345min', tz=None)
        expected_2 = DatetimeIndex(['2005-01-13 10:00:00', '2005-01-13 15:45:00'], dtype='datetime64[ns]', freq='345min', tz=None)
        tm.assert_index_equal(result_1, expected_1)
        tm.assert_index_equal(result_2, expected_2)

    def test_cached_range_bug(self):
        rng = date_range('2010-09-01 05:00:00', periods=50, freq=DateOffset(hours=6))
        assert len(rng) == 50
        assert rng[0] == datetime(2010, 9, 1, 5)

    def test_timezone_comparison_bug(self):
        start = Timestamp('20130220 10:00', tz='US/Eastern')
        result = date_range(start, periods=2, tz='US/Eastern')
        assert len(result) == 2

    def test_timezone_comparison_assert(self):
        start = Timestamp('20130220 10:00', tz='US/Eastern')
        msg = 'Inferred time zone not equal to passed time zone'
        with pytest.raises(AssertionError, match=msg):
            date_range(start, periods=2, tz='Europe/Berlin')

    def test_negative_non_tick_frequency_descending_dates(self, tz_aware_fixture):
        tz = tz_aware_fixture
        result = date_range(start='2011-06-01', end='2011-01-01', freq='-1MS', tz=tz)
        expected = date_range(end='2011-06-01', start='2011-01-01', freq='1MS', tz=tz)[::-1]
        tm.assert_index_equal(result, expected)

    def test_range_where_start_equal_end(self, inclusive_endpoints_fixture):
        start = '2021-09-02'
        end = '2021-09-02'
        result = date_range(start=start, end=end, freq='D', inclusive=inclusive_endpoints_fixture)
        both_range = date_range(start=start, end=end, freq='D', inclusive='both')
        if inclusive_endpoints_fixture == 'neither':
            expected = both_range[1:-1]
        elif inclusive_endpoints_fixture in ('left', 'right', 'both'):
            expected = both_range[:]
        tm.assert_index_equal(result, expected)

    def test_freq_dateoffset_with_relateivedelta_nanos(self):
        freq = DateOffset(hours=10, days=57, nanoseconds=3)
        result = date_range(end='1970-01-01 00:00:00', periods=10, freq=freq, name='a')
        expected = DatetimeIndex(['1968-08-02T05:59:59.999999973', '1968-09-28T15:59:59.999999976', '1968-11-25T01:59:59.999999979', '1969-01-21T11:59:59.999999982', '1969-03-19T21:59:59.999999985', '1969-05-16T07:59:59.999999988', '1969-07-12T17:59:59.999999991', '1969-09-08T03:59:59.999999994', '1969-11-04T13:59:59.999999997', '1970-01-01T00:00:00.000000000'], name='a')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('freq,freq_depr', [('h', 'H'), ('2min', '2T'), ('1s', '1S'), ('2ms', '2L'), ('1us', '1U'), ('2ns', '2N')])
    def test_frequencies_H_T_S_L_U_N_deprecated(self, freq, freq_depr):
        freq_msg = re.split('[0-9]*', freq, maxsplit=1)[1]
        freq_depr_msg = re.split('[0-9]*', freq_depr, maxsplit=1)[1]
        msg = f"'{freq_depr_msg}' is deprecated and will be removed in a future version, "
        f"please use '{freq_msg}' instead"
        expected = date_range('1/1/2000', periods=2, freq=freq)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = date_range('1/1/2000', periods=2, freq=freq_depr)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('freq,freq_depr', [('200YE', '200A'), ('YE', 'Y'), ('2YE-MAY', '2A-MAY'), ('YE-MAY', 'Y-MAY')])
    def test_frequencies_A_deprecated_Y_renamed(self, freq, freq_depr):
        freq_msg = re.split('[0-9]*', freq, maxsplit=1)[1]
        freq_depr_msg = re.split('[0-9]*', freq_depr, maxsplit=1)[1]
        msg = f"'{freq_depr_msg}' is deprecated and will be removed "
        f"in a future version, please use '{freq_msg}' instead."
        expected = date_range('1/1/2000', periods=2, freq=freq)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = date_range('1/1/2000', periods=2, freq=freq_depr)
        tm.assert_index_equal(result, expected)

    def test_to_offset_with_lowercase_deprecated_freq(self) -> None:
        msg = "'m' is deprecated and will be removed in a future version, please use 'ME' instead."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = date_range('2010-01-01', periods=2, freq='m')
        expected = DatetimeIndex(['2010-01-31', '2010-02-28'], freq='ME')
        tm.assert_index_equal(result, expected)

    def test_date_range_bday(self):
        sdate = datetime(1999, 12, 25)
        idx = date_range(start=sdate, freq='1B', periods=20)
        assert len(idx) == 20
        assert idx[0] == sdate + 0 * offsets.BDay()
        assert idx.freq == 'B'