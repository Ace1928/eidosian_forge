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
@pytest.mark.parametrize('series_length', [40, start_caching_at, start_caching_at + 1, start_caching_at + 5])
def test_to_datetime_cache_coerce_50_lines_outofbounds(series_length):
    ser = Series([datetime.fromisoformat('1446-04-12 00:00:00+00:00')] + [datetime.fromisoformat('1991-10-20 00:00:00+00:00')] * series_length, dtype=object)
    result1 = to_datetime(ser, errors='coerce', utc=True)
    expected1 = Series([NaT] + [Timestamp('1991-10-20 00:00:00+00:00')] * series_length)
    tm.assert_series_equal(result1, expected1)
    result2 = to_datetime(ser, errors='ignore', utc=True)
    expected2 = Series([datetime.fromisoformat('1446-04-12 00:00:00+00:00')] + [datetime.fromisoformat('1991-10-20 00:00:00+00:00')] * series_length)
    tm.assert_series_equal(result2, expected2)
    with pytest.raises(OutOfBoundsDatetime, match='Out of bounds nanosecond timestamp'):
        to_datetime(ser, errors='raise', utc=True)