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
@pytest.mark.parametrize('fmt,dates,expected_dates', [['%Y-%m-%d %H:%M:%S %Z', ['2010-01-01 12:00:00 UTC', '2010-01-01 12:00:00 GMT', '2010-01-01 12:00:00 US/Pacific'], [Timestamp('2010-01-01 12:00:00', tz='UTC'), Timestamp('2010-01-01 12:00:00', tz='GMT'), Timestamp('2010-01-01 12:00:00', tz='US/Pacific')]], ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 +0100', '2010-01-01 12:00:00 -0100'], [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60))), Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=-60)))]]])
def test_to_datetime_parse_tzname_or_tzoffset_utc_false_deprecated(self, fmt, dates, expected_dates):
    msg = 'parsing datetimes with mixed time zones will raise an error'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_datetime(dates, format=fmt)
    expected = Index(expected_dates)
    tm.assert_equal(result, expected)