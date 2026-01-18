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
@pytest.mark.parametrize('init_constructor, end_constructor', [(Index, DatetimeIndex), (list, DatetimeIndex), (np.array, DatetimeIndex), (Series, Series)])
def test_to_datetime_utc_true(self, cache, init_constructor, end_constructor):
    data = ['20100102 121314', '20100102 121315']
    expected_data = [Timestamp('2010-01-02 12:13:14', tz='utc'), Timestamp('2010-01-02 12:13:15', tz='utc')]
    result = to_datetime(init_constructor(data), format='%Y%m%d %H%M%S', utc=True, cache=cache)
    expected = end_constructor(expected_data)
    tm.assert_equal(result, expected)