from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
class TestWeekOfMonth:

    def test_constructor(self):
        with pytest.raises(ValueError, match='^Week'):
            WeekOfMonth(n=1, week=4, weekday=0)
        with pytest.raises(ValueError, match='^Week'):
            WeekOfMonth(n=1, week=-1, weekday=0)
        with pytest.raises(ValueError, match='^Day'):
            WeekOfMonth(n=1, week=0, weekday=-1)
        with pytest.raises(ValueError, match='^Day'):
            WeekOfMonth(n=1, week=0, weekday=-7)

    def test_repr(self):
        assert repr(WeekOfMonth(weekday=1, week=2)) == '<WeekOfMonth: week=2, weekday=1>'

    def test_offset(self):
        date1 = datetime(2011, 1, 4)
        date2 = datetime(2011, 1, 11)
        date3 = datetime(2011, 1, 18)
        date4 = datetime(2011, 1, 25)
        test_cases = [(-2, 2, 1, date1, datetime(2010, 11, 16)), (-2, 2, 1, date2, datetime(2010, 11, 16)), (-2, 2, 1, date3, datetime(2010, 11, 16)), (-2, 2, 1, date4, datetime(2010, 12, 21)), (-1, 2, 1, date1, datetime(2010, 12, 21)), (-1, 2, 1, date2, datetime(2010, 12, 21)), (-1, 2, 1, date3, datetime(2010, 12, 21)), (-1, 2, 1, date4, datetime(2011, 1, 18)), (0, 0, 1, date1, datetime(2011, 1, 4)), (0, 0, 1, date2, datetime(2011, 2, 1)), (0, 0, 1, date3, datetime(2011, 2, 1)), (0, 0, 1, date4, datetime(2011, 2, 1)), (0, 1, 1, date1, datetime(2011, 1, 11)), (0, 1, 1, date2, datetime(2011, 1, 11)), (0, 1, 1, date3, datetime(2011, 2, 8)), (0, 1, 1, date4, datetime(2011, 2, 8)), (0, 0, 1, date1, datetime(2011, 1, 4)), (0, 1, 1, date2, datetime(2011, 1, 11)), (0, 2, 1, date3, datetime(2011, 1, 18)), (0, 3, 1, date4, datetime(2011, 1, 25)), (1, 0, 0, date1, datetime(2011, 2, 7)), (1, 0, 0, date2, datetime(2011, 2, 7)), (1, 0, 0, date3, datetime(2011, 2, 7)), (1, 0, 0, date4, datetime(2011, 2, 7)), (1, 0, 1, date1, datetime(2011, 2, 1)), (1, 0, 1, date2, datetime(2011, 2, 1)), (1, 0, 1, date3, datetime(2011, 2, 1)), (1, 0, 1, date4, datetime(2011, 2, 1)), (1, 0, 2, date1, datetime(2011, 1, 5)), (1, 0, 2, date2, datetime(2011, 2, 2)), (1, 0, 2, date3, datetime(2011, 2, 2)), (1, 0, 2, date4, datetime(2011, 2, 2)), (1, 2, 1, date1, datetime(2011, 1, 18)), (1, 2, 1, date2, datetime(2011, 1, 18)), (1, 2, 1, date3, datetime(2011, 2, 15)), (1, 2, 1, date4, datetime(2011, 2, 15)), (2, 2, 1, date1, datetime(2011, 2, 15)), (2, 2, 1, date2, datetime(2011, 2, 15)), (2, 2, 1, date3, datetime(2011, 3, 15)), (2, 2, 1, date4, datetime(2011, 3, 15))]
        for n, week, weekday, dt, expected in test_cases:
            offset = WeekOfMonth(n, week=week, weekday=weekday)
            assert_offset_equal(offset, dt, expected)
        result = datetime(2011, 2, 1) - WeekOfMonth(week=1, weekday=2)
        assert result == datetime(2011, 1, 12)
        result = datetime(2011, 2, 3) - WeekOfMonth(week=0, weekday=2)
        assert result == datetime(2011, 2, 2)
    on_offset_cases = [(0, 0, datetime(2011, 2, 7), True), (0, 0, datetime(2011, 2, 6), False), (0, 0, datetime(2011, 2, 14), False), (1, 0, datetime(2011, 2, 14), True), (0, 1, datetime(2011, 2, 1), True), (0, 1, datetime(2011, 2, 8), False)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        week, weekday, dt, expected = case
        offset = WeekOfMonth(week=week, weekday=weekday)
        assert offset.is_on_offset(dt) == expected

    @pytest.mark.parametrize('n,week,date,tz', [(2, 2, '1916-05-15 01:14:49.583410462+0422', 'Asia/Qyzylorda'), (-3, 1, '1980-12-08 03:38:52.878321185+0500', 'Asia/Oral')])
    def test_is_on_offset_nanoseconds(self, n, week, date, tz):
        offset = WeekOfMonth(n=n, week=week, weekday=0)
        ts = Timestamp(date, tz=tz)
        fast = offset.is_on_offset(ts)
        slow = ts + offset - offset == ts
        assert fast == slow