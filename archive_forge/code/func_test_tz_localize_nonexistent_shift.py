from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('stamp, tz, forward_expected, backward_expected', [('2015-03-29 02:00:00', 'Europe/Warsaw', '2015-03-29 03:00:00', '2015-03-29 01:59:59'), ('2023-03-12 02:00:00', 'America/Los_Angeles', '2023-03-12 03:00:00', '2023-03-12 01:59:59'), ('2023-03-26 01:00:00', 'Europe/London', '2023-03-26 02:00:00', '2023-03-26 00:59:59'), ('2023-03-26 00:00:00', 'Atlantic/Azores', '2023-03-26 01:00:00', '2023-03-25 23:59:59')])
def test_tz_localize_nonexistent_shift(self, stamp, tz, forward_expected, backward_expected):
    ts = Timestamp(stamp)
    forward_ts = ts.tz_localize(tz, nonexistent='shift_forward')
    assert forward_ts == Timestamp(forward_expected, tz=tz)
    backward_ts = ts.tz_localize(tz, nonexistent='shift_backward')
    assert backward_ts == Timestamp(backward_expected, tz=tz)