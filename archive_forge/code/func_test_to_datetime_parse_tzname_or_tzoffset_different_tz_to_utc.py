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
def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(self):
    dates = ['2010-01-01 12:00:00 +0100', '2010-01-01 12:00:00 -0100', '2010-01-01 12:00:00 +0300', '2010-01-01 12:00:00 +0400']
    expected_dates = ['2010-01-01 11:00:00+00:00', '2010-01-01 13:00:00+00:00', '2010-01-01 09:00:00+00:00', '2010-01-01 08:00:00+00:00']
    fmt = '%Y-%m-%d %H:%M:%S %z'
    result = to_datetime(dates, format=fmt, utc=True)
    expected = DatetimeIndex(expected_dates)
    tm.assert_index_equal(result, expected)