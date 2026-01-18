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
def test_out_of_bounds_string(self):
    msg = "Cannot cast .* to unit='ns' without overflow"
    with pytest.raises(ValueError, match=msg):
        Timestamp('1676-01-01').as_unit('ns')
    with pytest.raises(ValueError, match=msg):
        Timestamp('2263-01-01').as_unit('ns')
    ts = Timestamp('2263-01-01')
    assert ts.unit == 's'
    ts = Timestamp('1676-01-01')
    assert ts.unit == 's'