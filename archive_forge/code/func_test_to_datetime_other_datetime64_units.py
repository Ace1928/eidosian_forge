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
def test_to_datetime_other_datetime64_units(self):
    scalar = np.int64(1337904000000000).view('M8[us]')
    as_obj = scalar.astype('O')
    index = DatetimeIndex([scalar])
    assert index[0] == scalar.astype('O')
    value = Timestamp(scalar)
    assert value == as_obj