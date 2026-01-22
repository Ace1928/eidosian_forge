from __future__ import annotations
from datetime import datetime
import pytest
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestBYearEnd:
    offset_cases = []
    offset_cases.append((BYearEnd(), {datetime(2008, 1, 1): datetime(2008, 12, 31), datetime(2008, 6, 30): datetime(2008, 12, 31), datetime(2008, 12, 31): datetime(2009, 12, 31), datetime(2005, 12, 30): datetime(2006, 12, 29), datetime(2005, 12, 31): datetime(2006, 12, 29)}))
    offset_cases.append((BYearEnd(0), {datetime(2008, 1, 1): datetime(2008, 12, 31), datetime(2008, 6, 30): datetime(2008, 12, 31), datetime(2008, 12, 31): datetime(2008, 12, 31), datetime(2005, 12, 31): datetime(2006, 12, 29)}))
    offset_cases.append((BYearEnd(-1), {datetime(2007, 1, 1): datetime(2006, 12, 29), datetime(2008, 6, 30): datetime(2007, 12, 31), datetime(2008, 12, 31): datetime(2007, 12, 31), datetime(2006, 12, 29): datetime(2005, 12, 30), datetime(2006, 12, 30): datetime(2006, 12, 29), datetime(2007, 1, 1): datetime(2006, 12, 29)}))
    offset_cases.append((BYearEnd(-2), {datetime(2007, 1, 1): datetime(2005, 12, 30), datetime(2008, 6, 30): datetime(2006, 12, 29), datetime(2008, 12, 31): datetime(2006, 12, 29)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
    on_offset_cases = [(BYearEnd(), datetime(2007, 12, 31), True), (BYearEnd(), datetime(2008, 1, 1), False), (BYearEnd(), datetime(2006, 12, 31), False), (BYearEnd(), datetime(2006, 12, 29), True)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)