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
def test_to_datetime_tz_psycopg2(self, request, cache):
    psycopg2_tz = pytest.importorskip('psycopg2.tz')
    tz1 = psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None)
    tz2 = psycopg2_tz.FixedOffsetTimezone(offset=-240, name=None)
    arr = np.array([datetime(2000, 1, 1, 3, 0, tzinfo=tz1), datetime(2000, 6, 1, 3, 0, tzinfo=tz2)], dtype=object)
    result = to_datetime(arr, errors='coerce', utc=True, cache=cache)
    expected = DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-06-01 07:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
    tm.assert_index_equal(result, expected)
    i = DatetimeIndex(['2000-01-01 08:00:00'], tz=psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None))
    assert is_datetime64_ns_dtype(i)
    result = to_datetime(i, errors='coerce', cache=cache)
    tm.assert_index_equal(result, i)
    result = to_datetime(i, errors='coerce', utc=True, cache=cache)
    expected = DatetimeIndex(['2000-01-01 13:00:00'], dtype='datetime64[ns, UTC]')
    tm.assert_index_equal(result, expected)