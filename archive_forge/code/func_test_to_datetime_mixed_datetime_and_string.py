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
def test_to_datetime_mixed_datetime_and_string(self):
    d1 = datetime(2020, 1, 1, 17, tzinfo=timezone(-timedelta(hours=1)))
    d2 = datetime(2020, 1, 1, 18, tzinfo=timezone(-timedelta(hours=1)))
    res = to_datetime(['2020-01-01 17:00 -0100', d2])
    expected = to_datetime([d1, d2]).tz_convert(timezone(timedelta(minutes=-60)))
    tm.assert_index_equal(res, expected)