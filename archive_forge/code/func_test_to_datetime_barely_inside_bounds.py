from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('timestamp', ['1677-09-21T00:12:43.145224193', '1677-09-21T00:12:43.145224999', '1677-09-21T00:12:43.145225000'])
def test_to_datetime_barely_inside_bounds(timestamp):
    result, _ = tslib.array_to_datetime(np.array([timestamp], dtype=object))
    tm.assert_numpy_array_equal(result, np.array([timestamp], dtype='M8[ns]'))