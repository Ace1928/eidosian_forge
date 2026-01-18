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
@pytest.mark.parametrize('freq', ['1D', '3D', '2ME', '7W', '3h', 'YE'])
def test_range_with_tz_closed_with_tz_aware_start_end(self, freq, inclusive_endpoints_fixture):
    begin = Timestamp('2011/1/1')
    end = Timestamp('2014/1/1')
    begintz = Timestamp('2011/1/1', tz='US/Eastern')
    endtz = Timestamp('2014/1/1', tz='US/Eastern')
    result_range = date_range(begin, end, inclusive=inclusive_endpoints_fixture, freq=freq, tz='US/Eastern')
    both_range = date_range(begin, end, inclusive='both', freq=freq, tz='US/Eastern')
    expected_range = _get_expected_range(begintz, endtz, both_range, inclusive_endpoints_fixture)
    tm.assert_index_equal(expected_range, result_range)