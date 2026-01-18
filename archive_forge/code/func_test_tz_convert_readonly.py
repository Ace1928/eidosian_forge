from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
def test_tz_convert_readonly():
    arr = np.array([0], dtype=np.int64)
    arr.setflags(write=False)
    result = tz_convert_from_utc(arr, UTC)
    tm.assert_numpy_array_equal(result, arr)