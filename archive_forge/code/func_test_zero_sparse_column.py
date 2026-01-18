import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_zero_sparse_column():
    df1 = pd.DataFrame({'A': SparseArray([0, 0, 0]), 'B': [1, 2, 3]})
    df2 = pd.DataFrame({'A': SparseArray([0, 1, 0]), 'B': [1, 2, 3]})
    result = df1.loc[df1['B'] != 2]
    expected = df2.loc[df2['B'] != 2]
    tm.assert_frame_equal(result, expected)
    expected = pd.DataFrame({'A': SparseArray([0, 0]), 'B': [1, 3]}, index=[0, 2])
    tm.assert_frame_equal(result, expected)