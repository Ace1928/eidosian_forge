from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import CDay
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_pickle_compat_0_14_1(self, datapath):
    hdays = [datetime(2013, 1, 1) for ele in range(4)]
    pth = datapath('tseries', 'offsets', 'data', 'cday-0.14.1.pickle')
    cday0_14_1 = read_pickle(pth)
    cday = CDay(holidays=hdays)
    assert cday == cday0_14_1