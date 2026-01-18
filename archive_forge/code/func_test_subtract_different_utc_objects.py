from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_subtract_different_utc_objects(self, utc_fixture, utc_fixture2):
    dt = datetime(2021, 1, 1)
    ts1 = Timestamp(dt, tz=utc_fixture)
    ts2 = Timestamp(dt, tz=utc_fixture2)
    result = ts1 - ts2
    expected = Timedelta(0)
    assert result == expected