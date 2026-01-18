import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_update_nan(self):
    df1 = DataFrame({'A': [1.0, 2, 3], 'B': date_range('2000', periods=3)})
    df2 = DataFrame({'A': [None, 2, 3]})
    expected = df1.copy()
    df1.update(df2, overwrite=False)
    tm.assert_frame_equal(df1, expected)
    df1 = DataFrame({'A': [1.0, None, 3], 'B': date_range('2000', periods=3)})
    df2 = DataFrame({'A': [None, 2, 3]})
    expected = DataFrame({'A': [1.0, 2, 3], 'B': date_range('2000', periods=3)})
    df1.update(df2, overwrite=False)
    tm.assert_frame_equal(df1, expected)