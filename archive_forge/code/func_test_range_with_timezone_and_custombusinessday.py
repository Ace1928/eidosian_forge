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
@pytest.mark.parametrize('start,period,expected', [('2022-07-23 00:00:00+02:00', 1, ['2022-07-25 00:00:00+02:00']), ('2022-07-22 00:00:00+02:00', 1, ['2022-07-22 00:00:00+02:00']), ('2022-07-22 00:00:00+02:00', 2, ['2022-07-22 00:00:00+02:00', '2022-07-25 00:00:00+02:00'])])
def test_range_with_timezone_and_custombusinessday(self, start, period, expected):
    result = date_range(start=start, periods=period, freq='C')
    expected = DatetimeIndex(expected).as_unit('ns')
    tm.assert_index_equal(result, expected)