import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_transpose_empty_preserves_datetimeindex(self):
    dti = DatetimeIndex([], dtype='M8[ns]')
    df = DataFrame(index=dti)
    expected = DatetimeIndex([], dtype='datetime64[ns]', freq=None)
    result1 = df.T.sum().index
    result2 = df.sum(axis=1).index
    tm.assert_index_equal(result1, expected)
    tm.assert_index_equal(result2, expected)