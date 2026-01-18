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
@pytest.mark.parametrize('tz', _tzs)
@pytest.mark.parametrize('ts_input,fold_out', [(1572136200000000000, 0), (1572139800000000000, 1), ('2019-10-27 01:30:00+01:00', 0), ('2019-10-27 01:30:00+00:00', 1), (datetime(2019, 10, 27, 1, 30, 0, 0, fold=0), 0), (datetime(2019, 10, 27, 1, 30, 0, 0, fold=1), 1)])
def test_timestamp_constructor_infer_fold_from_value(self, tz, ts_input, fold_out):
    ts = Timestamp(ts_input, tz=tz)
    result = ts.fold
    expected = fold_out
    assert result == expected