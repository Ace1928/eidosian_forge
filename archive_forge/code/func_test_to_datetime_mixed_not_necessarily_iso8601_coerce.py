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
@pytest.mark.parametrize(('errors', 'expected'), [('coerce', DatetimeIndex(['2020-01-01 00:00:00', NaT])), ('ignore', Index(['2020-01-01', '01-01-2000'], dtype=object))])
def test_to_datetime_mixed_not_necessarily_iso8601_coerce(errors, expected):
    result = to_datetime(['2020-01-01', '01-01-2000'], format='ISO8601', errors=errors)
    tm.assert_index_equal(result, expected)