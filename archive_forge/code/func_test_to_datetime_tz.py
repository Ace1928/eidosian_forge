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
def test_to_datetime_tz(self, cache):
    arr = [Timestamp('2013-01-01 13:00:00-0800', tz='US/Pacific'), Timestamp('2013-01-02 14:00:00-0800', tz='US/Pacific')]
    result = to_datetime(arr, cache=cache)
    expected = DatetimeIndex(['2013-01-01 13:00:00', '2013-01-02 14:00:00'], tz='US/Pacific')
    tm.assert_index_equal(result, expected)