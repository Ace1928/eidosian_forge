from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_timestamp_sub_datetime(self):
    dt = datetime(2013, 10, 12)
    ts = Timestamp(datetime(2013, 10, 13))
    assert (ts - dt).days == 1
    assert (dt - ts).days == -1