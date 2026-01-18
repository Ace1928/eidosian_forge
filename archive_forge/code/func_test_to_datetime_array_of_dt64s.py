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
@pytest.mark.parametrize('unit', ['s', 'D'])
def test_to_datetime_array_of_dt64s(self, cache, unit):
    dts = [np.datetime64('2000-01-01', unit), np.datetime64('2000-01-02', unit)] * 30
    result = to_datetime(dts, cache=cache)
    if cache:
        expected = DatetimeIndex([Timestamp(x).asm8 for x in dts], dtype='M8[s]')
    else:
        expected = DatetimeIndex([Timestamp(x).asm8 for x in dts], dtype='M8[ns]')
    tm.assert_index_equal(result, expected)
    dts_with_oob = dts + [np.datetime64('9999-01-01')]
    to_datetime(dts_with_oob, errors='raise')
    result = to_datetime(dts_with_oob, errors='coerce', cache=cache)
    if not cache:
        expected = DatetimeIndex([Timestamp(dts_with_oob[0]).asm8, Timestamp(dts_with_oob[1]).asm8] * 30 + [NaT])
    else:
        expected = DatetimeIndex(np.array(dts_with_oob, dtype='M8[s]'))
    tm.assert_index_equal(result, expected)
    result = to_datetime(dts_with_oob, errors='ignore', cache=cache)
    if not cache:
        expected = Index(dts_with_oob)
    tm.assert_index_equal(result, expected)