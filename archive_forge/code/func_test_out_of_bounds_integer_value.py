import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
def test_out_of_bounds_integer_value(self):
    msg = str(Timestamp.max._value * 2)
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        Timestamp(Timestamp.max._value * 2)
    msg = str(Timestamp.min._value * 2)
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        Timestamp(Timestamp.min._value * 2)