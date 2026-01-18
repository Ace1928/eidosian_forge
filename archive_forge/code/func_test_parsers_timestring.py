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
@pytest.mark.parametrize('date_str, exp_def', [['10:15', datetime(1, 1, 1, 10, 15)], ['9:05', datetime(1, 1, 1, 9, 5)]])
def test_parsers_timestring(self, date_str, exp_def):
    exp_now = parse(date_str)
    result1, _ = parsing.parse_datetime_string_with_reso(date_str)
    result2 = to_datetime(date_str)
    result3 = to_datetime([date_str])
    result4 = Timestamp(date_str)
    result5 = DatetimeIndex([date_str])[0]
    assert result1 == exp_def
    assert result2 == exp_now
    assert result3 == exp_now
    assert result4 == exp_now
    assert result5 == exp_now