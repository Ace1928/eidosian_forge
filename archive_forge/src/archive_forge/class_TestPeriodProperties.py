from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
class TestPeriodProperties:
    """Test properties such as year, month, weekday, etc...."""

    @pytest.mark.parametrize('freq', ['Y', 'M', 'D', 'h'])
    def test_is_leap_year(self, freq):
        p = Period('2000-01-01 00:00:00', freq=freq)
        assert p.is_leap_year
        assert isinstance(p.is_leap_year, bool)
        p = Period('1999-01-01 00:00:00', freq=freq)
        assert not p.is_leap_year
        p = Period('2004-01-01 00:00:00', freq=freq)
        assert p.is_leap_year
        p = Period('2100-01-01 00:00:00', freq=freq)
        assert not p.is_leap_year

    def test_quarterly_negative_ordinals(self):
        p = Period(ordinal=-1, freq='Q-DEC')
        assert p.year == 1969
        assert p.quarter == 4
        assert isinstance(p, Period)
        p = Period(ordinal=-2, freq='Q-DEC')
        assert p.year == 1969
        assert p.quarter == 3
        assert isinstance(p, Period)
        p = Period(ordinal=-2, freq='M')
        assert p.year == 1969
        assert p.month == 11
        assert isinstance(p, Period)

    def test_freq_str(self):
        i1 = Period('1982', freq='Min')
        assert i1.freq == offsets.Minute()
        assert i1.freqstr == 'min'

    @pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
    def test_period_deprecated_freq(self):
        cases = {'M': ['MTH', 'MONTH', 'MONTHLY', 'Mth', 'month', 'monthly'], 'B': ['BUS', 'BUSINESS', 'BUSINESSLY', 'WEEKDAY', 'bus'], 'D': ['DAY', 'DLY', 'DAILY', 'Day', 'Dly', 'Daily'], 'h': ['HR', 'HOUR', 'HRLY', 'HOURLY', 'hr', 'Hour', 'HRly'], 'min': ['minute', 'MINUTE', 'MINUTELY', 'minutely'], 's': ['sec', 'SEC', 'SECOND', 'SECONDLY', 'second'], 'ms': ['MILLISECOND', 'MILLISECONDLY', 'millisecond'], 'us': ['MICROSECOND', 'MICROSECONDLY', 'microsecond'], 'ns': ['NANOSECOND', 'NANOSECONDLY', 'nanosecond']}
        msg = INVALID_FREQ_ERR_MSG
        for exp, freqs in cases.items():
            for freq in freqs:
                with pytest.raises(ValueError, match=msg):
                    Period('2016-03-01 09:00', freq=freq)
                with pytest.raises(ValueError, match=msg):
                    Period(ordinal=1, freq=freq)
            p1 = Period('2016-03-01 09:00', freq=exp)
            p2 = Period(ordinal=1, freq=exp)
            assert isinstance(p1, Period)
            assert isinstance(p2, Period)

    @staticmethod
    def _period_constructor(bound, offset):
        return Period(year=bound.year, month=bound.month, day=bound.day, hour=bound.hour, minute=bound.minute, second=bound.second + offset, freq='us')

    @pytest.mark.parametrize('bound, offset', [(Timestamp.min, -1), (Timestamp.max, 1)])
    @pytest.mark.parametrize('period_property', ['start_time', 'end_time'])
    def test_outer_bounds_start_and_end_time(self, bound, offset, period_property):
        period = TestPeriodProperties._period_constructor(bound, offset)
        with pytest.raises(OutOfBoundsDatetime, match='Out of bounds nanosecond'):
            getattr(period, period_property)

    @pytest.mark.parametrize('bound, offset', [(Timestamp.min, -1), (Timestamp.max, 1)])
    @pytest.mark.parametrize('period_property', ['start_time', 'end_time'])
    def test_inner_bounds_start_and_end_time(self, bound, offset, period_property):
        period = TestPeriodProperties._period_constructor(bound, -offset)
        expected = period.to_timestamp().round(freq='s')
        assert getattr(period, period_property).round(freq='s') == expected
        expected = (bound - offset * Timedelta(1, unit='s')).floor('s')
        assert getattr(period, period_property).floor('s') == expected

    def test_start_time(self):
        freq_lst = ['Y', 'Q', 'M', 'D', 'h', 'min', 's']
        xp = datetime(2012, 1, 1)
        for f in freq_lst:
            p = Period('2012', freq=f)
            assert p.start_time == xp
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period('2012', freq='B').start_time == datetime(2012, 1, 2)
        assert Period('2012', freq='W').start_time == datetime(2011, 12, 26)

    def test_end_time(self):
        p = Period('2012', freq='Y')

        def _ex(*args):
            return Timestamp(Timestamp(datetime(*args)).as_unit('ns')._value - 1)
        xp = _ex(2013, 1, 1)
        assert xp == p.end_time
        p = Period('2012', freq='Q')
        xp = _ex(2012, 4, 1)
        assert xp == p.end_time
        p = Period('2012', freq='M')
        xp = _ex(2012, 2, 1)
        assert xp == p.end_time
        p = Period('2012', freq='D')
        xp = _ex(2012, 1, 2)
        assert xp == p.end_time
        p = Period('2012', freq='h')
        xp = _ex(2012, 1, 1, 1)
        assert xp == p.end_time
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            p = Period('2012', freq='B')
            xp = _ex(2012, 1, 3)
            assert xp == p.end_time
        p = Period('2012', freq='W')
        xp = _ex(2012, 1, 2)
        assert xp == p.end_time
        p = Period('2012', freq='15D')
        xp = _ex(2012, 1, 16)
        assert xp == p.end_time
        p = Period('2012', freq='1D1h')
        xp = _ex(2012, 1, 2, 1)
        assert xp == p.end_time
        p = Period('2012', freq='1h1D')
        xp = _ex(2012, 1, 2, 1)
        assert xp == p.end_time

    def test_end_time_business_friday(self):
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            per = Period('1990-01-05', 'B')
            result = per.end_time
        expected = Timestamp('1990-01-06') - Timedelta(nanoseconds=1)
        assert result == expected

    def test_anchor_week_end_time(self):

        def _ex(*args):
            return Timestamp(Timestamp(datetime(*args)).as_unit('ns')._value - 1)
        p = Period('2013-1-1', 'W-SAT')
        xp = _ex(2013, 1, 6)
        assert p.end_time == xp

    def test_properties_annually(self):
        a_date = Period(freq='Y', year=2007)
        assert a_date.year == 2007

    def test_properties_quarterly(self):
        qedec_date = Period(freq='Q-DEC', year=2007, quarter=1)
        qejan_date = Period(freq='Q-JAN', year=2007, quarter=1)
        qejun_date = Period(freq='Q-JUN', year=2007, quarter=1)
        for x in range(3):
            for qd in (qedec_date, qejan_date, qejun_date):
                assert (qd + x).qyear == 2007
                assert (qd + x).quarter == x + 1

    def test_properties_monthly(self):
        m_date = Period(freq='M', year=2007, month=1)
        for x in range(11):
            m_ival_x = m_date + x
            assert m_ival_x.year == 2007
            if 1 <= x + 1 <= 3:
                assert m_ival_x.quarter == 1
            elif 4 <= x + 1 <= 6:
                assert m_ival_x.quarter == 2
            elif 7 <= x + 1 <= 9:
                assert m_ival_x.quarter == 3
            elif 10 <= x + 1 <= 12:
                assert m_ival_x.quarter == 4
            assert m_ival_x.month == x + 1

    def test_properties_weekly(self):
        w_date = Period(freq='W', year=2007, month=1, day=7)
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31
        assert Period(freq='W', year=2012, month=2, day=1).days_in_month == 29

    def test_properties_weekly_legacy(self):
        w_date = Period(freq='W', year=2007, month=1, day=7)
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31
        exp = Period(freq='W', year=2012, month=2, day=1)
        assert exp.days_in_month == 29
        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            Period(freq='WK', year=2007, month=1, day=7)

    def test_properties_daily(self):
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            b_date = Period(freq='B', year=2007, month=1, day=1)
        assert b_date.year == 2007
        assert b_date.quarter == 1
        assert b_date.month == 1
        assert b_date.day == 1
        assert b_date.weekday == 0
        assert b_date.dayofyear == 1
        assert b_date.days_in_month == 31
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(freq='B', year=2012, month=2, day=1).days_in_month == 29
        d_date = Period(freq='D', year=2007, month=1, day=1)
        assert d_date.year == 2007
        assert d_date.quarter == 1
        assert d_date.month == 1
        assert d_date.day == 1
        assert d_date.weekday == 0
        assert d_date.dayofyear == 1
        assert d_date.days_in_month == 31
        assert Period(freq='D', year=2012, month=2, day=1).days_in_month == 29

    def test_properties_hourly(self):
        h_date1 = Period(freq='h', year=2007, month=1, day=1, hour=0)
        h_date2 = Period(freq='2h', year=2007, month=1, day=1, hour=0)
        for h_date in [h_date1, h_date2]:
            assert h_date.year == 2007
            assert h_date.quarter == 1
            assert h_date.month == 1
            assert h_date.day == 1
            assert h_date.weekday == 0
            assert h_date.dayofyear == 1
            assert h_date.hour == 0
            assert h_date.days_in_month == 31
            assert Period(freq='h', year=2012, month=2, day=1, hour=0).days_in_month == 29

    def test_properties_minutely(self):
        t_date = Period(freq='Min', year=2007, month=1, day=1, hour=0, minute=0)
        assert t_date.quarter == 1
        assert t_date.month == 1
        assert t_date.day == 1
        assert t_date.weekday == 0
        assert t_date.dayofyear == 1
        assert t_date.hour == 0
        assert t_date.minute == 0
        assert t_date.days_in_month == 31
        assert Period(freq='D', year=2012, month=2, day=1, hour=0, minute=0).days_in_month == 29

    def test_properties_secondly(self):
        s_date = Period(freq='Min', year=2007, month=1, day=1, hour=0, minute=0, second=0)
        assert s_date.year == 2007
        assert s_date.quarter == 1
        assert s_date.month == 1
        assert s_date.day == 1
        assert s_date.weekday == 0
        assert s_date.dayofyear == 1
        assert s_date.hour == 0
        assert s_date.minute == 0
        assert s_date.second == 0
        assert s_date.days_in_month == 31
        assert Period(freq='Min', year=2012, month=2, day=1, hour=0, minute=0, second=0).days_in_month == 29