import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
@pytest.mark.parametrize('fmt,dates,expected_dates', [['%Y-%m-%d %H:%M:%S %Z', ['2010-01-01 12:00:00 UTC'] * 2, [Timestamp('2010-01-01 12:00:00', tz='UTC')] * 2], ['%Y-%m-%d %H:%M:%S%z', ['2010-01-01 12:00:00+0100'] * 2, [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60)))] * 2], ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 +0100'] * 2, [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60)))] * 2], ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 Z', '2010-01-01 12:00:00 Z'], [Timestamp('2010-01-01 12:00:00', tzinfo=pytz.FixedOffset(0)), Timestamp('2010-01-01 12:00:00', tzinfo=pytz.FixedOffset(0))]]])
def test_to_datetime_parse_tzname_or_tzoffset(self, fmt, dates, expected_dates):
    result = to_datetime(dates, format=fmt)
    expected = Index(expected_dates)
    tm.assert_equal(result, expected)