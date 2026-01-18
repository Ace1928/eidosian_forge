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
def test_timestamp_constructor_invalid_fold_raise(self):
    msg = 'Valid values for the fold argument are None, 0, or 1.'
    with pytest.raises(ValueError, match=msg):
        Timestamp(123, fold=2)