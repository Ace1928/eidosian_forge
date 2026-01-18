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
def test_replace_out_of_pydatetime_bounds(self):
    ts = Timestamp('2016-01-01').as_unit('ns')
    msg = 'Out of bounds nanosecond timestamp: 99999-01-01 00:00:00'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        ts.replace(year=99999)
    ts = ts.as_unit('ms')
    result = ts.replace(year=99999)
    assert result.year == 99999
    assert result._value == Timestamp(np.datetime64('99999-01-01', 'ms'))._value