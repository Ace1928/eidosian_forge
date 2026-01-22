from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
class TestWeek:

    def test_repr(self):
        assert repr(Week(weekday=0)) == '<Week: weekday=0>'
        assert repr(Week(n=-1, weekday=0)) == '<-1 * Week: weekday=0>'
        assert repr(Week(n=-2, weekday=0)) == '<-2 * Weeks: weekday=0>'

    def test_corner(self):
        with pytest.raises(ValueError, match='Day must be'):
            Week(weekday=7)
        with pytest.raises(ValueError, match='Day must be'):
            Week(weekday=-1)

    def test_is_anchored(self):
        msg = 'Week.is_anchored is deprecated '
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert Week(weekday=0).is_anchored()
            assert not Week().is_anchored()
            assert not Week(2, weekday=2).is_anchored()
            assert not Week(2).is_anchored()
    offset_cases = []
    offset_cases.append((Week(), {datetime(2008, 1, 1): datetime(2008, 1, 8), datetime(2008, 1, 4): datetime(2008, 1, 11), datetime(2008, 1, 5): datetime(2008, 1, 12), datetime(2008, 1, 6): datetime(2008, 1, 13), datetime(2008, 1, 7): datetime(2008, 1, 14)}))
    offset_cases.append((Week(weekday=0), {datetime(2007, 12, 31): datetime(2008, 1, 7), datetime(2008, 1, 4): datetime(2008, 1, 7), datetime(2008, 1, 5): datetime(2008, 1, 7), datetime(2008, 1, 6): datetime(2008, 1, 7), datetime(2008, 1, 7): datetime(2008, 1, 14)}))
    offset_cases.append((Week(0, weekday=0), {datetime(2007, 12, 31): datetime(2007, 12, 31), datetime(2008, 1, 4): datetime(2008, 1, 7), datetime(2008, 1, 5): datetime(2008, 1, 7), datetime(2008, 1, 6): datetime(2008, 1, 7), datetime(2008, 1, 7): datetime(2008, 1, 7)}))
    offset_cases.append((Week(-2, weekday=1), {datetime(2010, 4, 6): datetime(2010, 3, 23), datetime(2010, 4, 8): datetime(2010, 3, 30), datetime(2010, 4, 5): datetime(2010, 3, 23)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize('weekday', range(7))
    def test_is_on_offset(self, weekday):
        offset = Week(weekday=weekday)
        for day in range(1, 8):
            date = datetime(2008, 1, day)
            expected = day % 7 == weekday
        assert_is_on_offset(offset, date, expected)

    @pytest.mark.parametrize('n,date', [(2, '1862-01-13 09:03:34.873477378+0210'), (-2, '1856-10-24 16:18:36.556360110-0717')])
    def test_is_on_offset_weekday_none(self, n, date):
        offset = Week(n=n, weekday=None)
        ts = Timestamp(date, tz='Africa/Lusaka')
        fast = offset.is_on_offset(ts)
        slow = ts + offset - offset == ts
        assert fast == slow

    def test_week_add_invalid(self):
        offset = Week(weekday=1)
        other = Day()
        with pytest.raises(TypeError, match='Cannot add'):
            offset + other