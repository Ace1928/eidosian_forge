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
def test_to_datetime_tz_mixed(self, cache):
    arr = [Timestamp('2013-01-01 13:00:00', tz='US/Pacific'), Timestamp('2013-01-02 14:00:00', tz='US/Eastern')]
    msg = 'Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True'
    with pytest.raises(ValueError, match=msg):
        to_datetime(arr, cache=cache)
    depr_msg = "errors='ignore' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = to_datetime(arr, cache=cache, errors='ignore')
    expected = Index([Timestamp('2013-01-01 13:00:00-08:00'), Timestamp('2013-01-02 14:00:00-05:00')], dtype='object')
    tm.assert_index_equal(result, expected)
    result = to_datetime(arr, cache=cache, errors='coerce')
    expected = DatetimeIndex(['2013-01-01 13:00:00-08:00', 'NaT'], dtype='datetime64[ns, US/Pacific]')
    tm.assert_index_equal(result, expected)