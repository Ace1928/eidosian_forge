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
def test_timestamp_to_datetime_tzoffset(self):
    tzinfo = tzoffset(None, 7200)
    expected = Timestamp('3/11/2012 04:00', tz=tzinfo)
    result = Timestamp(expected.to_pydatetime())
    assert expected == result