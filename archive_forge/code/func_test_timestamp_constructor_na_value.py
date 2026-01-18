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
@pytest.mark.parametrize('na_value', [None, np.nan, np.datetime64('NaT'), NaT, NA])
def test_timestamp_constructor_na_value(na_value):
    result = Timestamp(na_value)
    expected = NaT
    assert result is expected