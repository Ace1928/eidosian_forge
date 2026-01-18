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
def test_range_bug(self, unit):
    offset = DateOffset(months=3)
    result = date_range('2011-1-1', '2012-1-31', freq=offset, unit=unit)
    start = datetime(2011, 1, 1)
    expected = DatetimeIndex([start + i * offset for i in range(5)], dtype=f'M8[{unit}]', freq=offset)
    tm.assert_index_equal(result, expected)