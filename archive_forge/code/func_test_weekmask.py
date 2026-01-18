from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import CDay
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_weekmask(self):
    weekmask_saudi = 'Sat Sun Mon Tue Wed'
    weekmask_uae = '1111001'
    weekmask_egypt = [1, 1, 1, 1, 0, 0, 1]
    bday_saudi = CDay(weekmask=weekmask_saudi)
    bday_uae = CDay(weekmask=weekmask_uae)
    bday_egypt = CDay(weekmask=weekmask_egypt)
    dt = datetime(2013, 5, 1)
    xp_saudi = datetime(2013, 5, 4)
    xp_uae = datetime(2013, 5, 2)
    xp_egypt = datetime(2013, 5, 2)
    assert xp_saudi == dt + bday_saudi
    assert xp_uae == dt + bday_uae
    assert xp_egypt == dt + bday_egypt
    xp2 = datetime(2013, 5, 5)
    assert xp2 == dt + 2 * bday_saudi
    assert xp2 == dt + 2 * bday_uae
    assert xp2 == dt + 2 * bday_egypt