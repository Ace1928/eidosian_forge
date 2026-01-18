import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_update_filtered(self):
    df = DataFrame([[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]])
    other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])
    df.update(other, filter_func=lambda x: x > 2)
    expected = DataFrame([[1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]])
    tm.assert_frame_equal(df, expected)