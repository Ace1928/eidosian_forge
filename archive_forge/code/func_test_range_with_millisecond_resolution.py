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
@pytest.mark.parametrize('start_end', [('2018-01-01T00:00:01.000Z', '2018-01-03T00:00:01.000Z'), ('2018-01-01T00:00:00.010Z', '2018-01-03T00:00:00.010Z'), ('2001-01-01T00:00:00.010Z', '2001-01-03T00:00:00.010Z')])
def test_range_with_millisecond_resolution(self, start_end):
    start, end = start_end
    result = date_range(start=start, end=end, periods=2, inclusive='left')
    expected = DatetimeIndex([start], dtype='M8[ns, UTC]')
    tm.assert_index_equal(result, expected)