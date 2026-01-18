from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_tz_localize_utc_copies(self, utc_fixture):
    times = ['2015-03-08 01:00', '2015-03-08 02:00', '2015-03-08 03:00']
    index = DatetimeIndex(times)
    res = index.tz_localize(utc_fixture)
    assert not tm.shares_memory(res, index)
    res2 = index._data.tz_localize(utc_fixture)
    assert not tm.shares_memory(index._data, res2)