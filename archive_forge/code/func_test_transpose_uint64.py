import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_transpose_uint64(self):
    df = DataFrame({'A': np.arange(3), 'B': [2 ** 63, 2 ** 63 + 5, 2 ** 63 + 10]}, dtype=np.uint64)
    result = df.T
    expected = DataFrame(df.values.T)
    expected.index = ['A', 'B']
    tm.assert_frame_equal(result, expected)