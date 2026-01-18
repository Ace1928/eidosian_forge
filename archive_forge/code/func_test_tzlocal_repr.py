from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
def test_tzlocal_repr():
    ts = Timestamp('2011-01-01', tz=dateutil.tz.tzlocal())
    assert ts.tz == dateutil.tz.tzlocal()
    assert "tz='tzlocal()')" in repr(ts)