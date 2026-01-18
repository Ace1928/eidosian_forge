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
def test_to_datetime_unit_fractional_seconds(self):
    epoch = 1370745748
    ser = Series([epoch + t for t in np.arange(0, 2, 0.25)] + [iNaT]).astype(float)
    result = to_datetime(ser, unit='s')
    expected = Series([Timestamp('2013-06-09 02:42:28') + timedelta(seconds=t) for t in np.arange(0, 2, 0.25)] + [NaT], dtype='M8[ns]')
    result = result.round('ms')
    tm.assert_series_equal(result, expected)