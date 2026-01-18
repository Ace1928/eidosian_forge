from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_utc_box_timestamp_and_localize(self, tzstr):
    tz = timezones.maybe_get_tz(tzstr)
    rng = date_range('3/11/2012', '3/12/2012', freq='h', tz='utc')
    rng_eastern = rng.tz_convert(tzstr)
    expected = rng[-1].astimezone(tz)
    stamp = rng_eastern[-1]
    assert stamp == expected
    assert stamp.tzinfo == expected.tzinfo
    rng = date_range('3/13/2012', '3/14/2012', freq='h', tz='utc')
    rng_eastern = rng.tz_convert(tzstr)
    assert 'EDT' in repr(rng_eastern[0].tzinfo) or 'tzfile' in repr(rng_eastern[0].tzinfo)