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
@td.skip_if_windows
@pytest.mark.parametrize('tz', ['Pacific/Auckland', 'US/Samoa'])
def test_to_datetime_today(self, tz):
    with tm.set_timezone(tz):
        nptoday = np.datetime64('today').astype('datetime64[ns]').astype(np.int64)
        pdtoday = to_datetime('today')
        pdtoday2 = to_datetime(['today'])[0]
        tstoday = Timestamp('today').as_unit('ns')
        tstoday2 = Timestamp.today().as_unit('ns')
        assert abs(pdtoday.normalize()._value - nptoday) < 10000000000.0
        assert abs(pdtoday2.normalize()._value - nptoday) < 10000000000.0
        assert abs(pdtoday._value - tstoday._value) < 10000000000.0
        assert abs(pdtoday._value - tstoday2._value) < 10000000000.0
        assert pdtoday.tzinfo is None
        assert pdtoday2.tzinfo is None