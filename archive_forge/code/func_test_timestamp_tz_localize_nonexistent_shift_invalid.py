from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('offset', [-1, 1])
def test_timestamp_tz_localize_nonexistent_shift_invalid(self, offset, warsaw):
    tz = warsaw
    ts = Timestamp('2015-03-29 02:20:00')
    msg = 'The provided timedelta will relocalize on a nonexistent time'
    with pytest.raises(ValueError, match=msg):
        ts.tz_localize(tz, nonexistent=timedelta(seconds=offset))