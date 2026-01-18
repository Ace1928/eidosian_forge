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
@pytest.mark.parametrize('arg', ['2012-01-01 09:00:00.000000001', '2012-01-01 09:00:00.000001', '2012-01-01 09:00:00.001', '2012-01-01 09:00:00.001000', '2012-01-01 09:00:00.001000000'])
def test_parse_nanoseconds_with_formula(self, cache, arg):
    expected = to_datetime(arg, cache=cache)
    result = to_datetime(arg, format='%Y-%m-%d %H:%M:%S.%f', cache=cache)
    assert result == expected