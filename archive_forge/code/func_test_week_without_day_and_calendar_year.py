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
@pytest.mark.parametrize('date, format', [('2017-20', '%Y-%W'), ('20 Sunday', '%W %A'), ('20 Sun', '%W %a'), ('2017-21', '%Y-%U'), ('20 Sunday', '%U %A'), ('20 Sun', '%U %a')])
def test_week_without_day_and_calendar_year(self, date, format):
    msg = "Cannot use '%W' or '%U' without day and year"
    with pytest.raises(ValueError, match=msg):
        to_datetime(date, format=format)