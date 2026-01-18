import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
def test_weekday_but_no_day_raises(self):
    msg = 'Parsing datetimes with weekday but no day information is not supported'
    with pytest.raises(ValueError, match=msg):
        Timestamp('2023 Sept Thu')