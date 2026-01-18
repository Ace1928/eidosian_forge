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
@pytest.mark.parametrize('s, _format, dt', [['2015-1-1', '%G-%V-%u', datetime(2014, 12, 29, 0, 0)], ['2015-1-4', '%G-%V-%u', datetime(2015, 1, 1, 0, 0)], ['2015-1-7', '%G-%V-%u', datetime(2015, 1, 4, 0, 0)]])
def test_to_datetime_iso_week_year_format(self, s, _format, dt):
    assert to_datetime(s, format=_format) == dt