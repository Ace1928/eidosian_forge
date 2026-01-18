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
def test_string_na_nat_conversion(self, cache):
    strings = np.array(['1/1/2000', '1/2/2000', np.nan, '1/4/2000'], dtype=object)
    expected = np.empty(4, dtype='M8[ns]')
    for i, val in enumerate(strings):
        if isna(val):
            expected[i] = iNaT
        else:
            expected[i] = parse(val)
    result = tslib.array_to_datetime(strings)[0]
    tm.assert_almost_equal(result, expected)
    result2 = to_datetime(strings, cache=cache)
    assert isinstance(result2, DatetimeIndex)
    tm.assert_numpy_array_equal(result, result2.values)