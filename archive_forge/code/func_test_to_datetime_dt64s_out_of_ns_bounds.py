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
@pytest.mark.parametrize('dt', [np.datetime64('1000-01-01'), np.datetime64('5000-01-02')])
@pytest.mark.parametrize('errors', ['raise', 'ignore', 'coerce'])
def test_to_datetime_dt64s_out_of_ns_bounds(self, cache, dt, errors):
    ts = to_datetime(dt, errors=errors, cache=cache)
    assert isinstance(ts, Timestamp)
    assert ts.unit == 's'
    assert ts.asm8 == dt
    ts = Timestamp(dt)
    assert ts.unit == 's'
    assert ts.asm8 == dt