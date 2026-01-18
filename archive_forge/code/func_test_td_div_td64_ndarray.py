from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_div_td64_ndarray(self):
    td = Timedelta('1 day')
    other = np.array([Timedelta('2 Days').to_timedelta64()])
    expected = np.array([0.5])
    result = td / other
    tm.assert_numpy_array_equal(result, expected)
    result = other / td
    tm.assert_numpy_array_equal(result, expected * 4)