import numpy as np
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_to_numpy_mixed_dtype_to_str(self):
    df = DataFrame([[Timestamp('2020-01-01 00:00:00'), 100.0]])
    result = df.to_numpy(dtype=str)
    expected = np.array([['2020-01-01 00:00:00', '100.0']], dtype=str)
    tm.assert_numpy_array_equal(result, expected)