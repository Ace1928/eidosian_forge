from __future__ import annotations
from datetime import datetime
import pytest
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestBQuarterBegin:

    def test_repr(self):
        expected = '<BusinessQuarterBegin: startingMonth=3>'
        assert repr(BQuarterBegin()) == expected
        expected = '<BusinessQuarterBegin: startingMonth=3>'
        assert repr(BQuarterBegin(startingMonth=3)) == expected
        expected = '<BusinessQuarterBegin: startingMonth=1>'
        assert repr(BQuarterBegin(startingMonth=1)) == expected

    def test_is_anchored(self):
        msg = 'BQuarterBegin.is_anchored is deprecated '
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert BQuarterBegin(startingMonth=1).is_anchored()
            assert BQuarterBegin().is_anchored()
            assert not BQuarterBegin(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        offset = BQuarterBegin(n=-1, startingMonth=1)
        assert datetime(2007, 4, 3) + offset == datetime(2007, 4, 2)
    offset_cases = []
    offset_cases.append((BQuarterBegin(startingMonth=1), {datetime(2008, 1, 1): datetime(2008, 4, 1), datetime(2008, 1, 31): datetime(2008, 4, 1), datetime(2008, 2, 15): datetime(2008, 4, 1), datetime(2008, 2, 29): datetime(2008, 4, 1), datetime(2008, 3, 15): datetime(2008, 4, 1), datetime(2008, 3, 31): datetime(2008, 4, 1), datetime(2008, 4, 15): datetime(2008, 7, 1), datetime(2007, 3, 15): datetime(2007, 4, 2), datetime(2007, 2, 28): datetime(2007, 4, 2), datetime(2007, 1, 1): datetime(2007, 4, 2), datetime(2007, 4, 15): datetime(2007, 7, 2), datetime(2007, 7, 1): datetime(2007, 7, 2), datetime(2007, 4, 1): datetime(2007, 4, 2), datetime(2007, 4, 2): datetime(2007, 7, 2), datetime(2008, 4, 30): datetime(2008, 7, 1)}))
    offset_cases.append((BQuarterBegin(startingMonth=2), {datetime(2008, 1, 1): datetime(2008, 2, 1), datetime(2008, 1, 31): datetime(2008, 2, 1), datetime(2008, 1, 15): datetime(2008, 2, 1), datetime(2008, 2, 29): datetime(2008, 5, 1), datetime(2008, 3, 15): datetime(2008, 5, 1), datetime(2008, 3, 31): datetime(2008, 5, 1), datetime(2008, 4, 15): datetime(2008, 5, 1), datetime(2008, 8, 15): datetime(2008, 11, 3), datetime(2008, 9, 15): datetime(2008, 11, 3), datetime(2008, 11, 1): datetime(2008, 11, 3), datetime(2008, 4, 30): datetime(2008, 5, 1)}))
    offset_cases.append((BQuarterBegin(startingMonth=1, n=0), {datetime(2008, 1, 1): datetime(2008, 1, 1), datetime(2007, 12, 31): datetime(2008, 1, 1), datetime(2008, 2, 15): datetime(2008, 4, 1), datetime(2008, 2, 29): datetime(2008, 4, 1), datetime(2008, 1, 15): datetime(2008, 4, 1), datetime(2008, 2, 27): datetime(2008, 4, 1), datetime(2008, 3, 15): datetime(2008, 4, 1), datetime(2007, 4, 1): datetime(2007, 4, 2), datetime(2007, 4, 2): datetime(2007, 4, 2), datetime(2007, 7, 1): datetime(2007, 7, 2), datetime(2007, 4, 15): datetime(2007, 7, 2), datetime(2007, 7, 2): datetime(2007, 7, 2)}))
    offset_cases.append((BQuarterBegin(startingMonth=1, n=-1), {datetime(2008, 1, 1): datetime(2007, 10, 1), datetime(2008, 1, 31): datetime(2008, 1, 1), datetime(2008, 2, 15): datetime(2008, 1, 1), datetime(2008, 2, 29): datetime(2008, 1, 1), datetime(2008, 3, 15): datetime(2008, 1, 1), datetime(2008, 3, 31): datetime(2008, 1, 1), datetime(2008, 4, 15): datetime(2008, 4, 1), datetime(2007, 7, 3): datetime(2007, 7, 2), datetime(2007, 4, 3): datetime(2007, 4, 2), datetime(2007, 7, 2): datetime(2007, 4, 2), datetime(2008, 4, 1): datetime(2008, 1, 1)}))
    offset_cases.append((BQuarterBegin(startingMonth=1, n=2), {datetime(2008, 1, 1): datetime(2008, 7, 1), datetime(2008, 1, 15): datetime(2008, 7, 1), datetime(2008, 2, 29): datetime(2008, 7, 1), datetime(2008, 3, 15): datetime(2008, 7, 1), datetime(2007, 3, 31): datetime(2007, 7, 2), datetime(2007, 4, 15): datetime(2007, 10, 1), datetime(2008, 4, 30): datetime(2008, 10, 1)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)