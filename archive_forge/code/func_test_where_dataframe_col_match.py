from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_dataframe_col_match(self):
    df = DataFrame([[1, 2, 3], [4, 5, 6]])
    cond = DataFrame([[True, False, True], [False, False, True]])
    result = df.where(cond)
    expected = DataFrame([[1.0, np.nan, 3], [np.nan, np.nan, 6]])
    tm.assert_frame_equal(result, expected)
    cond.columns = ['a', 'b', 'c']
    result = df.where(cond)
    expected = DataFrame(np.nan, index=df.index, columns=df.columns)
    tm.assert_frame_equal(result, expected)