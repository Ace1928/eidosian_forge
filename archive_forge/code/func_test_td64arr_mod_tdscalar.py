from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_mod_tdscalar(self, box_with_array, three_days):
    tdi = timedelta_range('1 Day', '9 days')
    tdarr = tm.box_expected(tdi, box_with_array)
    expected = TimedeltaIndex(['1 Day', '2 Days', '0 Days'] * 3)
    expected = tm.box_expected(expected, box_with_array)
    result = tdarr % three_days
    tm.assert_equal(result, expected)
    warn = None
    if box_with_array is DataFrame and isinstance(three_days, pd.DateOffset):
        warn = PerformanceWarning
        expected = expected.astype(object)
    with tm.assert_produces_warning(warn):
        result = divmod(tdarr, three_days)
    tm.assert_equal(result[1], expected)
    tm.assert_equal(result[0], tdarr // three_days)