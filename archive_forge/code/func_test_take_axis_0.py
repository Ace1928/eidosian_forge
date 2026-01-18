from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_take_axis_0(self):
    arr = np.arange(12).reshape(4, 3)
    result = algos.take(arr, [0, -1])
    expected = np.array([[0, 1, 2], [9, 10, 11]])
    tm.assert_numpy_array_equal(result, expected)
    result = algos.take(arr, [0, -1], allow_fill=True, fill_value=0)
    expected = np.array([[0, 1, 2], [0, 0, 0]])
    tm.assert_numpy_array_equal(result, expected)