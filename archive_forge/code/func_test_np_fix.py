from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
@td.skip_copy_on_write_not_yet_implemented
def test_np_fix():
    ser = pd.Series([-1.5, -0.5, 0.5, 1.5])
    result = np.fix(ser)
    expected = pd.Series([-1.0, -0.0, 0.0, 1.0])
    tm.assert_series_equal(result, expected)