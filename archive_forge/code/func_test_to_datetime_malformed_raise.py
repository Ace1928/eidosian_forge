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
def test_to_datetime_malformed_raise(self):
    ts_strings = ['200622-12-31', '111111-24-11']
    msg = 'Parsed string "200622-12-31" gives an invalid tzoffset, which must be between -timedelta\\(hours=24\\) and timedelta\\(hours=24\\), at position 0'
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(UserWarning, match='Could not infer format'):
            to_datetime(ts_strings, errors='raise')