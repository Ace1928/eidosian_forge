from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import CDay
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_weekmask_and_holidays(self):
    weekmask_egypt = 'Sun Mon Tue Wed Thu'
    holidays = ['2012-05-01', datetime(2013, 5, 1), np.datetime64('2014-05-01')]
    bday_egypt = CDay(holidays=holidays, weekmask=weekmask_egypt)
    dt = datetime(2013, 4, 30)
    xp_egypt = datetime(2013, 5, 5)
    assert xp_egypt == dt + 2 * bday_egypt