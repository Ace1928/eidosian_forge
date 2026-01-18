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
@pytest.mark.parametrize('timestamp', ['2018-01-01 0:0:0.124999360', '2018-01-01 0:0:0.125000367', '2018-01-01 0:0:0.125500', '2018-01-01 0:0:0.126500', '2018-01-01 12:00:00', '2019-01-01 12:00:00'])
@pytest.mark.parametrize('freq', ['2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '250ns', '500ns', '750ns', '1us', '19us', '250us', '500us', '750us', '1s', '2s', '3s', '1D'])
def test_round_int64(self, timestamp, freq):
    dt = Timestamp(timestamp).as_unit('ns')
    unit = to_offset(freq).nanos
    result = dt.floor(freq)
    assert result._value % unit == 0, f'floor not a {freq} multiple'
    assert 0 <= dt._value - result._value < unit, 'floor error'
    result = dt.ceil(freq)
    assert result._value % unit == 0, f'ceil not a {freq} multiple'
    assert 0 <= result._value - dt._value < unit, 'ceil error'
    result = dt.round(freq)
    assert result._value % unit == 0, f'round not a {freq} multiple'
    assert abs(result._value - dt._value) <= unit // 2, 'round error'
    if unit % 2 == 0 and abs(result._value - dt._value) == unit // 2:
        assert result._value // unit % 2 == 0, 'round half to even error'