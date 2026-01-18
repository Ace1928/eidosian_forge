from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_add_sub_dt64_ndarray(self):
    td = Timedelta('1 day')
    other = np.array(['2000-01-01'], dtype='M8[ns]')
    expected = np.array(['2000-01-02'], dtype='M8[ns]')
    tm.assert_numpy_array_equal(td + other, expected)
    tm.assert_numpy_array_equal(other + td, expected)
    expected = np.array(['1999-12-31'], dtype='M8[ns]')
    tm.assert_numpy_array_equal(-td + other, expected)
    tm.assert_numpy_array_equal(other - td, expected)