from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
def test_np_matmul():
    df1 = pd.DataFrame(data=[[-1, 1, 10]])
    df2 = pd.DataFrame(data=[-1, 1, 10])
    expected = pd.DataFrame(data=[102])
    result = np.matmul(df1, df2)
    tm.assert_frame_equal(expected, result)