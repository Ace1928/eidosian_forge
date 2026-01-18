from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_add_sub_td64_ndarray(self):
    td = Timedelta('1 day')
    other = np.array([td.to_timedelta64()])
    expected = np.array([Timedelta('2 Days').to_timedelta64()])
    result = td + other
    tm.assert_numpy_array_equal(result, expected)
    result = other + td
    tm.assert_numpy_array_equal(result, expected)
    result = td - other
    tm.assert_numpy_array_equal(result, expected * 0)
    result = other - td
    tm.assert_numpy_array_equal(result, expected * 0)