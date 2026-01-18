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
@pytest.mark.parametrize('tz', ['dateutil/Europe/London', None])
@pytest.mark.parametrize('fold', [0, 1])
def test_timestamp_constructor_retain_fold(self, tz, fold):
    ts = Timestamp(year=2019, month=10, day=27, hour=1, minute=30, tz=tz, fold=fold)
    result = ts.fold
    expected = fold
    assert result == expected