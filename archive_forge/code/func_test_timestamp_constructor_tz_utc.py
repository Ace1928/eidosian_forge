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
def test_timestamp_constructor_tz_utc(self):
    utc_stamp = Timestamp('3/11/2012 05:00', tz='utc')
    assert utc_stamp.tzinfo is timezone.utc
    assert utc_stamp.hour == 5
    utc_stamp = Timestamp('3/11/2012 05:00').tz_localize('utc')
    assert utc_stamp.hour == 5