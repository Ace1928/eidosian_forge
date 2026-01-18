from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [operator.add, ops.radd])
def test_td_add_mixed_timedeltalike_object_dtype_array(self, op):
    now = Timestamp('2021-11-09 09:54:00')
    arr = np.array([now, Timedelta('1D')])
    exp = np.array([now + Timedelta('1D'), Timedelta('2D')])
    res = op(arr, Timedelta('1D'))
    tm.assert_numpy_array_equal(res, exp)