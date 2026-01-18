from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('data,expected', [(['01-01-2013', '01-02-2013'], ['2013-01-01T00:00:00.000000000', '2013-01-02T00:00:00.000000000']), (['Mon Sep 16 2013', 'Tue Sep 17 2013'], ['2013-09-16T00:00:00.000000000', '2013-09-17T00:00:00.000000000'])])
def test_parsing_valid_dates(data, expected):
    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr)
    expected = np.array(expected, dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)