from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
def _compare_utc_to_local(tz_didx):

    def f(x):
        return tzconversion.tz_convert_from_utc_single(x, tz_didx.tz)
    result = tz_convert_from_utc(tz_didx.asi8, tz_didx.tz)
    expected = np.vectorize(f)(tz_didx.asi8)
    tm.assert_numpy_array_equal(result, expected)