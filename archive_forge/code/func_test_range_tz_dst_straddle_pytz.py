from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
@pytest.mark.parametrize('start, end', [[Timestamp(datetime(2014, 3, 6), tz='US/Eastern'), Timestamp(datetime(2014, 3, 12), tz='US/Eastern')], [Timestamp(datetime(2013, 11, 1), tz='US/Eastern'), Timestamp(datetime(2013, 11, 6), tz='US/Eastern')]])
def test_range_tz_dst_straddle_pytz(self, start, end):
    dr = date_range(start, end, freq='D')
    assert dr[0] == start
    assert dr[-1] == end
    assert np.all(dr.hour == 0)
    dr = date_range(start, end, freq='D', tz='US/Eastern')
    assert dr[0] == start
    assert dr[-1] == end
    assert np.all(dr.hour == 0)
    dr = date_range(start.replace(tzinfo=None), end.replace(tzinfo=None), freq='D', tz='US/Eastern')
    assert dr[0] == start
    assert dr[-1] == end
    assert np.all(dr.hour == 0)