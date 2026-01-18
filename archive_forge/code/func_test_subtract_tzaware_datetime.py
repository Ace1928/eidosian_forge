from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_subtract_tzaware_datetime(self):
    t1 = Timestamp('2020-10-22T22:00:00+00:00')
    t2 = datetime(2020, 10, 22, 22, tzinfo=timezone.utc)
    result = t1 - t2
    assert isinstance(result, Timedelta)
    assert result == Timedelta('0 days')