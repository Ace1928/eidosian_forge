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
def test_timestamp_constructor_pytz_fold_raise(self):
    msg = 'pytz timezones do not support fold. Please use dateutil timezones.'
    tz = pytz.timezone('Europe/London')
    with pytest.raises(ValueError, match=msg):
        Timestamp(datetime(2019, 10, 27, 0, 30, 0, 0), tz=tz, fold=0)