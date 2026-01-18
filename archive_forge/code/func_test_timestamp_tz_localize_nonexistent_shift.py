from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('start_ts, tz, end_ts, shift', [['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:00:00', 'forward'], ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:59:59.999999999', 'backward'], ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:20:00', timedelta(hours=1)], ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:20:00', timedelta(hours=-1)], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:00:00', 'forward'], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:59:59.999999999', 'backward'], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:33:00', timedelta(hours=1)], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:33:00', timedelta(hours=-1)]])
@pytest.mark.parametrize('tz_type', ['', 'dateutil/'])
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_timestamp_tz_localize_nonexistent_shift(self, start_ts, tz, end_ts, shift, tz_type, unit):
    tz = tz_type + tz
    if isinstance(shift, str):
        shift = 'shift_' + shift
    ts = Timestamp(start_ts).as_unit(unit)
    result = ts.tz_localize(tz, nonexistent=shift)
    expected = Timestamp(end_ts).tz_localize(tz)
    if unit == 'us':
        assert result == expected.replace(nanosecond=0)
    elif unit == 'ms':
        micros = expected.microsecond - expected.microsecond % 1000
        assert result == expected.replace(microsecond=micros, nanosecond=0)
    elif unit == 's':
        assert result == expected.replace(microsecond=0, nanosecond=0)
    else:
        assert result == expected
    assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value