from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
def test_parsing_different_timezone_offsets():
    data = ['2015-11-18 15:30:00+05:30', '2015-11-18 15:30:00+06:30']
    data = np.array(data, dtype=object)
    msg = 'parsing datetimes with mixed time zones will raise an error'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result, result_tz = tslib.array_to_datetime(data)
    expected = np.array([datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 19800)), datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 23400))], dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    assert result_tz is None