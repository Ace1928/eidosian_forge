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
@pytest.mark.parametrize('arg, expected, format', [['1/1/2000', '20000101', '%d/%m/%Y'], ['1/1/2000', '20000101', '%m/%d/%Y'], ['1/2/2000', '20000201', '%d/%m/%Y'], ['1/2/2000', '20000102', '%m/%d/%Y'], ['1/3/2000', '20000301', '%d/%m/%Y'], ['1/3/2000', '20000103', '%m/%d/%Y']])
def test_to_datetime_format_scalar(self, cache, arg, expected, format):
    result = to_datetime(arg, format=format, cache=cache)
    expected = Timestamp(expected)
    assert result == expected