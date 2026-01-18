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
def test_to_datetime_inconsistent_format(self, cache):
    data = ['01/01/2011 00:00:00', '01-02-2011 00:00:00', '2011-01-03T00:00:00']
    ser = Series(np.array(data))
    msg = f"""^time data "01-02-2011 00:00:00" doesn\\'t match format "%m/%d/%Y %H:%M:%S", at position 1. {PARSING_ERR_MSG}$"""
    with pytest.raises(ValueError, match=msg):
        to_datetime(ser, cache=cache)