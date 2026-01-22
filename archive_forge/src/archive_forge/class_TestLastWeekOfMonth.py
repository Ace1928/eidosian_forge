from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
class TestLastWeekOfMonth:

    def test_constructor(self):
        with pytest.raises(ValueError, match='^N cannot be 0'):
            LastWeekOfMonth(n=0, weekday=1)
        with pytest.raises(ValueError, match='^Day'):
            LastWeekOfMonth(n=1, weekday=-1)
        with pytest.raises(ValueError, match='^Day'):
            LastWeekOfMonth(n=1, weekday=7)

    def test_offset(self):
        last_sat = datetime(2013, 8, 31)
        next_sat = datetime(2013, 9, 28)
        offset_sat = LastWeekOfMonth(n=1, weekday=5)
        one_day_before = last_sat + timedelta(days=-1)
        assert one_day_before + offset_sat == last_sat
        one_day_after = last_sat + timedelta(days=+1)
        assert one_day_after + offset_sat == next_sat
        assert last_sat + offset_sat == next_sat
        offset_thur = LastWeekOfMonth(n=1, weekday=3)
        last_thurs = datetime(2013, 1, 31)
        next_thurs = datetime(2013, 2, 28)
        one_day_before = last_thurs + timedelta(days=-1)
        assert one_day_before + offset_thur == last_thurs
        one_day_after = last_thurs + timedelta(days=+1)
        assert one_day_after + offset_thur == next_thurs
        assert last_thurs + offset_thur == next_thurs
        three_before = last_thurs + timedelta(days=-3)
        assert three_before + offset_thur == last_thurs
        two_after = last_thurs + timedelta(days=+2)
        assert two_after + offset_thur == next_thurs
        offset_sunday = LastWeekOfMonth(n=1, weekday=WeekDay.SUN)
        assert datetime(2013, 7, 31) + offset_sunday == datetime(2013, 8, 25)
    on_offset_cases = [(WeekDay.SUN, datetime(2013, 1, 27), True), (WeekDay.SAT, datetime(2013, 3, 30), True), (WeekDay.MON, datetime(2013, 2, 18), False), (WeekDay.SUN, datetime(2013, 2, 25), False), (WeekDay.MON, datetime(2013, 2, 25), True), (WeekDay.SAT, datetime(2013, 11, 30), True), (WeekDay.SAT, datetime(2006, 8, 26), True), (WeekDay.SAT, datetime(2007, 8, 25), True), (WeekDay.SAT, datetime(2008, 8, 30), True), (WeekDay.SAT, datetime(2009, 8, 29), True), (WeekDay.SAT, datetime(2010, 8, 28), True), (WeekDay.SAT, datetime(2011, 8, 27), True), (WeekDay.SAT, datetime(2019, 8, 31), True)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        weekday, dt, expected = case
        offset = LastWeekOfMonth(weekday=weekday)
        assert offset.is_on_offset(dt) == expected

    @pytest.mark.parametrize('n,weekday,date,tz', [(4, 6, '1917-05-27 20:55:27.084284178+0200', 'Europe/Warsaw'), (-4, 5, '2005-08-27 05:01:42.799392561-0500', 'America/Rainy_River')])
    def test_last_week_of_month_on_offset(self, n, weekday, date, tz):
        offset = LastWeekOfMonth(n=n, weekday=weekday)
        ts = Timestamp(date, tz=tz)
        slow = ts + offset - offset == ts
        fast = offset.is_on_offset(ts)
        assert fast == slow

    def test_repr(self):
        assert repr(LastWeekOfMonth(n=2, weekday=1)) == '<2 * LastWeekOfMonths: weekday=1>'