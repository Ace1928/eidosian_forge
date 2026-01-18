from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_us_federal_holiday_with_datetime(self):
    bhour_us = CustomBusinessHour(calendar=USFederalHolidayCalendar())
    t0 = datetime(2014, 1, 17, 15)
    result = t0 + bhour_us * 8
    expected = Timestamp('2014-01-21 15:00:00')
    assert result == expected