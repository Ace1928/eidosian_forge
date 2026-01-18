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
def test_to_datetime_np_str(self):
    value = np.str_('2019-02-04 10:18:46.297000+0000')
    ser = Series([value])
    exp = Timestamp('2019-02-04 10:18:46.297000', tz='UTC')
    assert to_datetime(value) == exp
    assert to_datetime(ser.iloc[0]) == exp
    res = to_datetime([value])
    expected = Index([exp])
    tm.assert_index_equal(res, expected)
    res = to_datetime(ser)
    expected = Series(expected)
    tm.assert_series_equal(res, expected)