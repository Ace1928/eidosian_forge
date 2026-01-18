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
@td.skip_if_windows
def test_to_datetime_now(self):
    with tm.set_timezone('US/Eastern'):
        now = Timestamp('now').as_unit('ns')
        pdnow = to_datetime('now')
        pdnow2 = to_datetime(['now'])[0]
        assert abs(pdnow._value - now._value) < 10000000000.0
        assert abs(pdnow2._value - now._value) < 10000000000.0
        assert pdnow.tzinfo is None
        assert pdnow2.tzinfo is None