from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rdiv_ndarray(self):
    td = Timedelta(10, unit='d')
    arr = np.array([td], dtype=object)
    result = arr / td
    expected = np.array([1], dtype=np.float64)
    tm.assert_numpy_array_equal(result, expected)
    arr = np.array([None])
    result = arr / td
    expected = np.array([np.nan])
    tm.assert_numpy_array_equal(result, expected)
    arr = np.array([np.nan], dtype=object)
    msg = "unsupported operand type\\(s\\) for /: 'float' and 'Timedelta'"
    with pytest.raises(TypeError, match=msg):
        arr / td
    arr = np.array([np.nan], dtype=np.float64)
    msg = 'cannot use operands with types dtype'
    with pytest.raises(TypeError, match=msg):
        arr / td