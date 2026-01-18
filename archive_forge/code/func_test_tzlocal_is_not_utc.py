from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
def test_tzlocal_is_not_utc():
    tz = dateutil.tz.tzlocal()
    assert not timezones.is_utc(tz)
    assert not timezones.tz_compare(tz, dateutil.tz.tzutc())