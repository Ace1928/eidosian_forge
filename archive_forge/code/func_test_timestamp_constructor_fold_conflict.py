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
@pytest.mark.parametrize('fold', [0, 1])
@pytest.mark.parametrize('ts_input', [1572136200000000000, 1.5721362e+18, np.datetime64(1572136200000000000, 'ns'), '2019-10-27 01:30:00+01:00', datetime(2019, 10, 27, 0, 30, 0, 0, tzinfo=timezone.utc)])
def test_timestamp_constructor_fold_conflict(self, ts_input, fold):
    msg = 'Cannot pass fold with possibly unambiguous input: int, float, numpy.datetime64, str, or timezone-aware datetime-like. Pass naive datetime-like or build Timestamp from components.'
    with pytest.raises(ValueError, match=msg):
        Timestamp(ts_input=ts_input, fold=fold)