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
@pytest.mark.parametrize('value,fmt,expected', [['2009324', '%Y%W%w', Timestamp('2009-08-13')], ['2013020', '%Y%U%w', Timestamp('2013-01-13')]])
def test_to_datetime_format_weeks(self, value, fmt, expected, cache):
    assert to_datetime(value, format=fmt, cache=cache) == expected