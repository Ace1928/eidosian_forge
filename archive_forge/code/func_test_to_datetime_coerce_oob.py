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
@pytest.mark.parametrize('string_arg, format', [('March 1, 2018', '%B %d, %Y'), ('2018-03-01', '%Y-%m-%d')])
@pytest.mark.parametrize('outofbounds', [datetime(9999, 1, 1), date(9999, 1, 1), np.datetime64('9999-01-01'), 'January 1, 9999', '9999-01-01'])
def test_to_datetime_coerce_oob(self, string_arg, format, outofbounds):
    ts_strings = [string_arg, outofbounds]
    result = to_datetime(ts_strings, errors='coerce', format=format)
    expected = DatetimeIndex([datetime(2018, 3, 1), NaT])
    tm.assert_index_equal(result, expected)