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
def test_to_datetime_format_YYYYMMDD_with_nat(self, cache):
    ser = Series([19801222, 19801222] + [19810105] * 5, dtype='float')
    expected = Series([Timestamp('19801222'), Timestamp('19801222')] + [Timestamp('19810105')] * 5)
    expected[2] = np.nan
    ser[2] = np.nan
    result = to_datetime(ser, format='%Y%m%d', cache=cache)
    tm.assert_series_equal(result, expected)
    ser2 = ser.apply(str)
    ser2[2] = 'nat'
    with pytest.raises(ValueError, match='unconverted data remains when parsing with format "%Y%m%d": ".0", at position 0'):
        to_datetime(ser2, format='%Y%m%d', cache=cache)