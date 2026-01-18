from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_bug(self):
    df = DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [4.0, 3.0, 2.0, 1.0]}, dtype='float64')
    expected = DataFrame({'a': [np.nan, np.nan, 3.0, 4.0], 'b': [4.0, 3.0, np.nan, np.nan]}, dtype='float64')
    result = df.where(df > 2, np.nan)
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    return_value = result.where(result > 2, np.nan, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(result, expected)