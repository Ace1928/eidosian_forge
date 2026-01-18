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
def test_to_datetime_format_microsecond(self, cache):
    month_abbr = calendar.month_abbr[4]
    val = f'01-{month_abbr}-2011 00:00:01.978'
    format = '%d-%b-%Y %H:%M:%S.%f'
    result = to_datetime(val, format=format, cache=cache)
    exp = datetime.strptime(val, format)
    assert result == exp