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
@pytest.mark.parametrize('test_input, freq, expected', [('2018-01-01 00:02:06', '2s', '2018-01-01 00:02:06'), ('2018-01-01 00:02:00', '2T', '2018-01-01 00:02:00'), ('2018-01-01 00:04:00', '4T', '2018-01-01 00:04:00'), ('2018-01-01 00:15:00', '15T', '2018-01-01 00:15:00'), ('2018-01-01 00:20:00', '20T', '2018-01-01 00:20:00'), ('2018-01-01 03:00:00', '3H', '2018-01-01 03:00:00')])
@pytest.mark.parametrize('rounder', ['ceil', 'floor', 'round'])
def test_round_minute_freq(self, test_input, freq, expected, rounder):
    dt = Timestamp(test_input)
    expected = Timestamp(expected)
    func = getattr(dt, rounder)
    result = func(freq)
    assert result == expected