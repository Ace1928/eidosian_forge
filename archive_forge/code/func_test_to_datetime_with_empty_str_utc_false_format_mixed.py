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
def test_to_datetime_with_empty_str_utc_false_format_mixed():
    vals = ['2020-01-01 00:00+00:00', '']
    result = to_datetime(vals, format='mixed')
    expected = Index([Timestamp('2020-01-01 00:00+00:00'), 'NaT'], dtype='M8[ns, UTC]')
    tm.assert_index_equal(result, expected)
    alt = to_datetime(vals)
    tm.assert_index_equal(alt, expected)
    alt2 = DatetimeIndex(vals)
    tm.assert_index_equal(alt2, expected)