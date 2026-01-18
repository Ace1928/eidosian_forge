from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [pytz.timezone('US/Eastern'), gettz('US/Eastern'), 'US/Eastern', 'dateutil/US/Eastern'])
def test_timestamp_add_timedelta_push_over_dst_boundary(self, tz):
    stamp = Timestamp('3/10/2012 22:00', tz=tz)
    result = stamp + timedelta(hours=6)
    expected = Timestamp('3/11/2012 05:00', tz=tz)
    assert result == expected