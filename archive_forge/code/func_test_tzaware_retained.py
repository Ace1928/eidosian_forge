import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.util import cartesian_product
def test_tzaware_retained(self):
    x = date_range('2000-01-01', periods=2, tz='US/Pacific')
    y = np.array([3, 4])
    result1, result2 = cartesian_product([x, y])
    expected = x.repeat(2)
    tm.assert_index_equal(result1, expected)