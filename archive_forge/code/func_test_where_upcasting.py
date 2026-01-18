from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_upcasting(self):
    df = DataFrame({c: Series([1] * 3, dtype=c) for c in ['float32', 'float64', 'int32', 'int64']})
    df.iloc[1, :] = 0
    result = df.dtypes
    expected = Series([np.dtype('float32'), np.dtype('float64'), np.dtype('int32'), np.dtype('int64')], index=['float32', 'float64', 'int32', 'int64'])
    tm.assert_series_equal(result, expected)