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
def test_unit_rounding(self, cache):
    value = 1434743731.877
    result = to_datetime(value, unit='s', cache=cache)
    expected = Timestamp('2015-06-19 19:55:31.877000093')
    assert result == expected
    alt = Timestamp(value, unit='s')
    assert alt == result