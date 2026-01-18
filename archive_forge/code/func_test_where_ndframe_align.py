from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_ndframe_align(self):
    msg = 'Array conditional must be same shape as self'
    df = DataFrame([[1, 2, 3], [4, 5, 6]])
    cond = [True]
    with pytest.raises(ValueError, match=msg):
        df.where(cond)
    expected = DataFrame([[1, 2, 3], [np.nan, np.nan, np.nan]])
    out = df.where(Series(cond))
    tm.assert_frame_equal(out, expected)
    cond = np.array([False, True, False, True])
    with pytest.raises(ValueError, match=msg):
        df.where(cond)
    expected = DataFrame([[np.nan, np.nan, np.nan], [4, 5, 6]])
    out = df.where(Series(cond))
    tm.assert_frame_equal(out, expected)