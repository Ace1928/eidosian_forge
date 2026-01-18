from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
def test_tzlocal_offset():
    ts = Timestamp('2011-01-01', tz=dateutil.tz.tzlocal())
    offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))
    offset = offset.total_seconds()
    assert ts._value + offset == Timestamp('2011-01-01')._value