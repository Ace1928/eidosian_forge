from __future__ import annotations
from datetime import datetime
import pytest
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestBQuarterEnd:

    def test_repr(self):
        expected = '<BusinessQuarterEnd: startingMonth=3>'
        assert repr(BQuarterEnd()) == expected
        expected = '<BusinessQuarterEnd: startingMonth=3>'
        assert repr(BQuarterEnd(startingMonth=3)) == expected
        expected = '<BusinessQuarterEnd: startingMonth=1>'
        assert repr(BQuarterEnd(startingMonth=1)) == expected

    def test_is_anchored(self):
        msg = 'BQuarterEnd.is_anchored is deprecated '
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert BQuarterEnd(startingMonth=1).is_anchored()
            assert BQuarterEnd().is_anchored()
            assert not BQuarterEnd(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        offset = BQuarterEnd(n=-1, startingMonth=1)
        assert datetime(2010, 1, 31) + offset == datetime(2010, 1, 29)
    offset_cases = []
    offset_cases.append((BQuarterEnd(startingMonth=1), {datetime(2008, 1, 1): datetime(2008, 1, 31), datetime(2008, 1, 31): datetime(2008, 4, 30), datetime(2008, 2, 15): datetime(2008, 4, 30), datetime(2008, 2, 29): datetime(2008, 4, 30), datetime(2008, 3, 15): datetime(2008, 4, 30), datetime(2008, 3, 31): datetime(2008, 4, 30), datetime(2008, 4, 15): datetime(2008, 4, 30), datetime(2008, 4, 30): datetime(2008, 7, 31)}))
    offset_cases.append((BQuarterEnd(startingMonth=2), {datetime(2008, 1, 1): datetime(2008, 2, 29), datetime(2008, 1, 31): datetime(2008, 2, 29), datetime(2008, 2, 15): datetime(2008, 2, 29), datetime(2008, 2, 29): datetime(2008, 5, 30), datetime(2008, 3, 15): datetime(2008, 5, 30), datetime(2008, 3, 31): datetime(2008, 5, 30), datetime(2008, 4, 15): datetime(2008, 5, 30), datetime(2008, 4, 30): datetime(2008, 5, 30)}))
    offset_cases.append((BQuarterEnd(startingMonth=1, n=0), {datetime(2008, 1, 1): datetime(2008, 1, 31), datetime(2008, 1, 31): datetime(2008, 1, 31), datetime(2008, 2, 15): datetime(2008, 4, 30), datetime(2008, 2, 29): datetime(2008, 4, 30), datetime(2008, 3, 15): datetime(2008, 4, 30), datetime(2008, 3, 31): datetime(2008, 4, 30), datetime(2008, 4, 15): datetime(2008, 4, 30), datetime(2008, 4, 30): datetime(2008, 4, 30)}))
    offset_cases.append((BQuarterEnd(startingMonth=1, n=-1), {datetime(2008, 1, 1): datetime(2007, 10, 31), datetime(2008, 1, 31): datetime(2007, 10, 31), datetime(2008, 2, 15): datetime(2008, 1, 31), datetime(2008, 2, 29): datetime(2008, 1, 31), datetime(2008, 3, 15): datetime(2008, 1, 31), datetime(2008, 3, 31): datetime(2008, 1, 31), datetime(2008, 4, 15): datetime(2008, 1, 31), datetime(2008, 4, 30): datetime(2008, 1, 31)}))
    offset_cases.append((BQuarterEnd(startingMonth=1, n=2), {datetime(2008, 1, 31): datetime(2008, 7, 31), datetime(2008, 2, 15): datetime(2008, 7, 31), datetime(2008, 2, 29): datetime(2008, 7, 31), datetime(2008, 3, 15): datetime(2008, 7, 31), datetime(2008, 3, 31): datetime(2008, 7, 31), datetime(2008, 4, 15): datetime(2008, 7, 31), datetime(2008, 4, 30): datetime(2008, 10, 31)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
    on_offset_cases = [(BQuarterEnd(1, startingMonth=1), datetime(2008, 1, 31), True), (BQuarterEnd(1, startingMonth=1), datetime(2007, 12, 31), False), (BQuarterEnd(1, startingMonth=1), datetime(2008, 2, 29), False), (BQuarterEnd(1, startingMonth=1), datetime(2007, 3, 30), False), (BQuarterEnd(1, startingMonth=1), datetime(2007, 3, 31), False), (BQuarterEnd(1, startingMonth=1), datetime(2008, 4, 30), True), (BQuarterEnd(1, startingMonth=1), datetime(2008, 5, 30), False), (BQuarterEnd(1, startingMonth=1), datetime(2007, 6, 29), False), (BQuarterEnd(1, startingMonth=1), datetime(2007, 6, 30), False), (BQuarterEnd(1, startingMonth=2), datetime(2008, 1, 31), False), (BQuarterEnd(1, startingMonth=2), datetime(2007, 12, 31), False), (BQuarterEnd(1, startingMonth=2), datetime(2008, 2, 29), True), (BQuarterEnd(1, startingMonth=2), datetime(2007, 3, 30), False), (BQuarterEnd(1, startingMonth=2), datetime(2007, 3, 31), False), (BQuarterEnd(1, startingMonth=2), datetime(2008, 4, 30), False), (BQuarterEnd(1, startingMonth=2), datetime(2008, 5, 30), True), (BQuarterEnd(1, startingMonth=2), datetime(2007, 6, 29), False), (BQuarterEnd(1, startingMonth=2), datetime(2007, 6, 30), False), (BQuarterEnd(1, startingMonth=3), datetime(2008, 1, 31), False), (BQuarterEnd(1, startingMonth=3), datetime(2007, 12, 31), True), (BQuarterEnd(1, startingMonth=3), datetime(2008, 2, 29), False), (BQuarterEnd(1, startingMonth=3), datetime(2007, 3, 30), True), (BQuarterEnd(1, startingMonth=3), datetime(2007, 3, 31), False), (BQuarterEnd(1, startingMonth=3), datetime(2008, 4, 30), False), (BQuarterEnd(1, startingMonth=3), datetime(2008, 5, 30), False), (BQuarterEnd(1, startingMonth=3), datetime(2007, 6, 29), True), (BQuarterEnd(1, startingMonth=3), datetime(2007, 6, 30), False)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)