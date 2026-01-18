from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
def test_tz_localize_ambiguous_compat(self):
    naive = Timestamp('2013-10-27 01:00:00')
    pytz_zone = 'Europe/London'
    dateutil_zone = 'dateutil/Europe/London'
    result_pytz = naive.tz_localize(pytz_zone, ambiguous=False)
    result_dateutil = naive.tz_localize(dateutil_zone, ambiguous=False)
    assert result_pytz._value == result_dateutil._value
    assert result_pytz._value == 1382835600
    assert result_pytz.to_pydatetime().tzname() == 'GMT'
    assert result_dateutil.to_pydatetime().tzname() == 'GMT'
    assert str(result_pytz) == str(result_dateutil)
    result_pytz = naive.tz_localize(pytz_zone, ambiguous=True)
    result_dateutil = naive.tz_localize(dateutil_zone, ambiguous=True)
    assert result_pytz._value == result_dateutil._value
    assert result_pytz._value == 1382832000
    assert str(result_pytz) == str(result_dateutil)
    assert result_pytz.to_pydatetime().tzname() == result_dateutil.to_pydatetime().tzname()