from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_mul_numeric_ndarray_0d(self):
    td = Timedelta('1 day')
    other = np.array(2)
    assert other.ndim == 0
    expected = Timedelta('2 days')
    res = td * other
    assert type(res) is Timedelta
    assert res == expected
    res = other * td
    assert type(res) is Timedelta
    assert res == expected