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
@pytest.mark.parametrize('format, expected', [['%d/%m/%Y', [Timestamp('20000101'), Timestamp('20000201'), Timestamp('20000301')]], ['%m/%d/%Y', [Timestamp('20000101'), Timestamp('20000102'), Timestamp('20000103')]]])
def test_to_datetime_format(self, cache, index_or_series, format, expected):
    values = index_or_series(['1/1/2000', '1/2/2000', '1/3/2000'])
    result = to_datetime(values, format=format, cache=cache)
    expected = index_or_series(expected)
    tm.assert_equal(result, expected)