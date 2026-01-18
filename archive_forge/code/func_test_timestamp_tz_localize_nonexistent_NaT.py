from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_timestamp_tz_localize_nonexistent_NaT(self, warsaw, unit):
    tz = warsaw
    ts = Timestamp('2015-03-29 02:20:00').as_unit(unit)
    result = ts.tz_localize(tz, nonexistent='NaT')
    assert result is NaT