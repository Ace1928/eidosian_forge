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
def test_to_datetime_consistent_format(self, cache):
    data = ['Jan/01/2011', 'Feb/01/2011', 'Mar/01/2011']
    ser = Series(np.array(data))
    result = to_datetime(ser, cache=cache)
    expected = Series(['2011-01-01', '2011-02-01', '2011-03-01'], dtype='datetime64[ns]')
    tm.assert_series_equal(result, expected)