from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
def test_tz_localize_ambiguous_raise(self):
    ts = Timestamp('2015-11-1 01:00')
    msg = 'Cannot infer dst time from 2015-11-01 01:00:00,'
    with pytest.raises(AmbiguousTimeError, match=msg):
        ts.tz_localize('US/Pacific', ambiguous='raise')