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
@pytest.mark.parametrize('tz, normalize', [(pytz.timezone('US/Eastern'), lambda x: x.tzinfo.normalize(x)), (gettz('US/Eastern'), lambda x: x)])
def test_replace_across_dst(self, tz, normalize):
    ts_naive = Timestamp('2017-12-03 16:03:30')
    ts_aware = conversion.localize_pydatetime(ts_naive, tz)
    assert ts_aware == normalize(ts_aware)
    ts2 = ts_aware.replace(month=6)
    assert (ts2.hour, ts2.minute) == (ts_aware.hour, ts_aware.minute)
    ts2b = normalize(ts2)
    assert ts2 == ts2b