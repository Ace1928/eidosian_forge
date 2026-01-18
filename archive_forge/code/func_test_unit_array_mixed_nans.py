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
def test_unit_array_mixed_nans(self, cache):
    values = [11111111111111111, 1, 1.0, iNaT, NaT, np.nan, 'NaT', '']
    result = to_datetime(values, unit='D', errors='ignore', cache=cache)
    expected = Index([11111111111111111, Timestamp('1970-01-02'), Timestamp('1970-01-02'), NaT, NaT, NaT, NaT, NaT], dtype=object)
    tm.assert_index_equal(result, expected)
    result = to_datetime(values, unit='D', errors='coerce', cache=cache)
    expected = DatetimeIndex(['NaT', '1970-01-02', '1970-01-02', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT'], dtype='M8[ns]')
    tm.assert_index_equal(result, expected)
    msg = "cannot convert input 11111111111111111 with the unit 'D'"
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        to_datetime(values, unit='D', errors='raise', cache=cache)