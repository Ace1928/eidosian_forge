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
@pytest.mark.parametrize('scalar, expected', [['20100102 121314', Timestamp('2010-01-02 12:13:14', tz='utc')], ['20100102 121315', Timestamp('2010-01-02 12:13:15', tz='utc')]])
def test_to_datetime_utc_true_scalar(self, cache, scalar, expected):
    result = to_datetime(scalar, format='%Y%m%d %H%M%S', utc=True, cache=cache)
    assert result == expected