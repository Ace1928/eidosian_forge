from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_rmod_tdscalar(self, box_with_array, three_days):
    tdi = timedelta_range('1 Day', '9 days')
    tdarr = tm.box_expected(tdi, box_with_array)
    expected = ['0 Days', '1 Day', '0 Days'] + ['3 Days'] * 6
    expected = TimedeltaIndex(expected)
    expected = tm.box_expected(expected, box_with_array)
    result = three_days % tdarr
    tm.assert_equal(result, expected)
    result = divmod(three_days, tdarr)
    tm.assert_equal(result[1], expected)
    tm.assert_equal(result[0], three_days // tdarr)