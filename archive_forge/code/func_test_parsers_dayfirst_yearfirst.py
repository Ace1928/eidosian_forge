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
@pytest.mark.parametrize('date_str, dayfirst, yearfirst, expected', [('10-11-12', False, False, datetime(2012, 10, 11)), ('10-11-12', True, False, datetime(2012, 11, 10)), ('10-11-12', False, True, datetime(2010, 11, 12)), ('10-11-12', True, True, datetime(2010, 12, 11)), ('20/12/21', False, False, datetime(2021, 12, 20)), ('20/12/21', True, False, datetime(2021, 12, 20)), ('20/12/21', False, True, datetime(2020, 12, 21)), ('20/12/21', True, True, datetime(2020, 12, 21))])
def test_parsers_dayfirst_yearfirst(self, cache, date_str, dayfirst, yearfirst, expected):
    dateutil_result = parse(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
    assert dateutil_result == expected
    result1, _ = parsing.parse_datetime_string_with_reso(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
    if not dayfirst and (not yearfirst):
        result2 = Timestamp(date_str)
        assert result2 == expected
    result3 = to_datetime(date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache)
    result4 = DatetimeIndex([date_str], dayfirst=dayfirst, yearfirst=yearfirst)[0]
    assert result1 == expected
    assert result3 == expected
    assert result4 == expected