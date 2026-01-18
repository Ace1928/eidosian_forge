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
@pytest.mark.parametrize('format', [None, '%Y-%m-%d'])
def test_to_datetime_iso8601_noleading_0s(self, cache, format):
    ser = Series(['2014-1-1', '2014-2-2', '2015-3-3'])
    expected = Series([Timestamp('2014-01-01'), Timestamp('2014-02-02'), Timestamp('2015-03-03')])
    result = to_datetime(ser, format=format, cache=cache)
    tm.assert_series_equal(result, expected)