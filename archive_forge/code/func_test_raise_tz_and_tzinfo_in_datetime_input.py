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
@pytest.mark.parametrize('box', [datetime, Timestamp])
def test_raise_tz_and_tzinfo_in_datetime_input(self, box):
    kwargs = {'year': 2018, 'month': 1, 'day': 1, 'tzinfo': pytz.utc}
    msg = 'Cannot pass a datetime or Timestamp'
    with pytest.raises(ValueError, match=msg):
        Timestamp(box(**kwargs), tz='US/Pacific')
    msg = 'Cannot pass a datetime or Timestamp'
    with pytest.raises(ValueError, match=msg):
        Timestamp(box(**kwargs), tzinfo=pytz.timezone('US/Pacific'))