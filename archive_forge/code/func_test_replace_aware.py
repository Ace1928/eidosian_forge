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
def test_replace_aware(self, tz_aware_fixture):
    tz = tz_aware_fixture
    ts = Timestamp('2016-01-01 09:00:00', tz=tz)
    result = ts.replace(hour=0)
    expected = Timestamp('2016-01-01 00:00:00', tz=tz)
    assert result == expected