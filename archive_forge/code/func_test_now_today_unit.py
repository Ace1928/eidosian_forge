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
@pytest.mark.parametrize('method', ['now', 'today'])
def test_now_today_unit(self, method):
    ts_from_method = getattr(Timestamp, method)()
    ts_from_string = Timestamp(method)
    assert ts_from_method.unit == ts_from_string.unit == 'us'