from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
def test_tz_localize_nonexistent_invalid_arg(self, warsaw):
    tz = warsaw
    ts = Timestamp('2015-03-29 02:00:00')
    msg = "The nonexistent argument must be one of 'raise', 'NaT', 'shift_forward', 'shift_backward' or a timedelta object"
    with pytest.raises(ValueError, match=msg):
        ts.tz_localize(tz, nonexistent='foo')