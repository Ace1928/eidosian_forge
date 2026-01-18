from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('data', [['-352.737091', '183.575577'], ['1', '2', '3', '4', '5']])
def test_number_looking_strings_not_into_datetime(data):
    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr, errors='ignore')
    tm.assert_numpy_array_equal(result, arr)