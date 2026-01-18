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
def test_timestamp_constructor_near_dst_boundary(self):
    for tz in ['Europe/Brussels', 'Europe/Prague']:
        result = Timestamp('2015-10-25 01:00', tz=tz)
        expected = Timestamp('2015-10-25 01:00').tz_localize(tz)
        assert result == expected
        msg = 'Cannot infer dst time from 2015-10-25 02:00:00'
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            Timestamp('2015-10-25 02:00', tz=tz)
    result = Timestamp('2017-03-26 01:00', tz='Europe/Paris')
    expected = Timestamp('2017-03-26 01:00').tz_localize('Europe/Paris')
    assert result == expected
    msg = '2017-03-26 02:00'
    with pytest.raises(pytz.NonExistentTimeError, match=msg):
        Timestamp('2017-03-26 02:00', tz='Europe/Paris')
    naive = Timestamp('2015-11-18 10:00:00')
    result = naive.tz_localize('UTC').tz_convert('Asia/Kolkata')
    expected = Timestamp('2015-11-18 15:30:00+0530', tz='Asia/Kolkata')
    assert result == expected
    result = Timestamp('2017-03-26 00:00', tz='Europe/Paris')
    expected = Timestamp('2017-03-26 00:00:00+0100', tz='Europe/Paris')
    assert result == expected
    result = Timestamp('2017-03-26 01:00', tz='Europe/Paris')
    expected = Timestamp('2017-03-26 01:00:00+0100', tz='Europe/Paris')
    assert result == expected
    msg = '2017-03-26 02:00'
    with pytest.raises(pytz.NonExistentTimeError, match=msg):
        Timestamp('2017-03-26 02:00', tz='Europe/Paris')
    result = Timestamp('2017-03-26 02:00:00+0100', tz='Europe/Paris')
    naive = Timestamp(result.as_unit('ns')._value)
    expected = naive.tz_localize('UTC').tz_convert('Europe/Paris')
    assert result == expected
    result = Timestamp('2017-03-26 03:00', tz='Europe/Paris')
    expected = Timestamp('2017-03-26 03:00:00+0200', tz='Europe/Paris')
    assert result == expected