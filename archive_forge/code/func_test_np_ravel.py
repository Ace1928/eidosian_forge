import numpy as np
from pandas import (
import pandas._testing as tm
def test_np_ravel(self):
    arr = np.array([[0.11197053, 0.44361564, -0.92589452], [0.05883648, -0.00948922, -0.26469934]])
    result = np.ravel([DataFrame(batch.reshape(1, 3)) for batch in arr])
    expected = np.array([0.11197053, 0.44361564, -0.92589452, 0.05883648, -0.00948922, -0.26469934])
    tm.assert_numpy_array_equal(result, expected)
    result = np.ravel(DataFrame(arr[0].reshape(1, 3), columns=['x1', 'x2', 'x3']))
    expected = np.array([0.11197053, 0.44361564, -0.92589452])
    tm.assert_numpy_array_equal(result, expected)
    result = np.ravel([DataFrame(batch.reshape(1, 3), columns=['x1', 'x2', 'x3']) for batch in arr])
    expected = np.array([0.11197053, 0.44361564, -0.92589452, 0.05883648, -0.00948922, -0.26469934])
    tm.assert_numpy_array_equal(result, expected)