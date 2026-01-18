from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
@pytest.mark.parametrize('method, ts_str, freq', [['ceil', '2018-03-11 01:59:00-0600', '5min'], ['round', '2018-03-11 01:59:00-0600', '5min'], ['floor', '2018-03-11 03:01:00-0500', '2H']])
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_round_dst_border_nonexistent(self, method, ts_str, freq, unit):
    ts = Timestamp(ts_str, tz='America/Chicago').as_unit(unit)
    result = getattr(ts, method)(freq, nonexistent='shift_forward')
    expected = Timestamp('2018-03-11 03:00:00', tz='America/Chicago')
    assert result == expected
    assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
    result = getattr(ts, method)(freq, nonexistent='NaT')
    assert result is NaT
    msg = '2018-03-11 02:00:00'
    with pytest.raises(pytz.NonExistentTimeError, match=msg):
        getattr(ts, method)(freq, nonexistent='raise')