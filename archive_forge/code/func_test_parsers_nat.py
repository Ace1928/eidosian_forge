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
def test_parsers_nat(self):
    result1, _ = parsing.parse_datetime_string_with_reso('NaT')
    result2 = to_datetime('NaT')
    result3 = Timestamp('NaT')
    result4 = DatetimeIndex(['NaT'])[0]
    assert result1 is NaT
    assert result2 is NaT
    assert result3 is NaT
    assert result4 is NaT