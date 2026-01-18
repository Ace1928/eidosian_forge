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
def test_to_datetime_unit_na_values(self):
    result = to_datetime([1, 2, 'NaT', NaT, np.nan], unit='D')
    expected = DatetimeIndex([Timestamp('1970-01-02'), Timestamp('1970-01-03')] + ['NaT'] * 3, dtype='M8[ns]')
    tm.assert_index_equal(result, expected)