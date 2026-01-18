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
@pytest.mark.parametrize('arg, format', [('2001-01-01', '%Y-%m-%d'), ('01-01-2001', '%d-%m-%Y')])
def test_to_datetime_dt64s_and_str(self, arg, format):
    result = to_datetime([arg, np.datetime64('2020-01-01')], format=format)
    expected = DatetimeIndex(['2001-01-01', '2020-01-01'])
    tm.assert_index_equal(result, expected)