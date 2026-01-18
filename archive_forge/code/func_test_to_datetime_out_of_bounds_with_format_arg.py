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
@pytest.mark.parametrize('format, warning', [(None, UserWarning), ('%Y-%m-%d %H:%M:%S', None), ('%Y-%d-%m %H:%M:%S', None)])
def test_to_datetime_out_of_bounds_with_format_arg(self, format, warning):
    msg = '^Out of bounds nanosecond timestamp: 2417-10-10 00:00:00, at position 0'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        to_datetime('2417-10-10 00:00:00', format=format)