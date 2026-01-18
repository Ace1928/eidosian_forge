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
@pytest.mark.parametrize('unit', ['Y', 'M'])
@pytest.mark.parametrize('item', [150, float(150)])
def test_to_datetime_month_or_year_unit_int(self, cache, unit, item, request):
    ts = Timestamp(item, unit=unit)
    expected = DatetimeIndex([ts], dtype='M8[ns]')
    result = to_datetime([item], unit=unit, cache=cache)
    tm.assert_index_equal(result, expected)
    result = to_datetime(np.array([item], dtype=object), unit=unit, cache=cache)
    tm.assert_index_equal(result, expected)
    result = to_datetime(np.array([item]), unit=unit, cache=cache)
    tm.assert_index_equal(result, expected)
    result = to_datetime(np.array([item, np.nan]), unit=unit, cache=cache)
    assert result.isna()[1]
    tm.assert_index_equal(result[:1], expected)