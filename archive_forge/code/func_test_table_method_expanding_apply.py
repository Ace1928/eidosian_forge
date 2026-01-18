import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_table_method_expanding_apply(self, axis, nogil, parallel, nopython):
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}

    def f(x):
        return np.sum(x, axis=0) + 1
    df = DataFrame(np.eye(3))
    result = df.expanding(method='table', axis=axis).apply(f, raw=True, engine_kwargs=engine_kwargs, engine='numba')
    expected = df.expanding(method='single', axis=axis).apply(f, raw=True, engine_kwargs=engine_kwargs, engine='numba')
    tm.assert_frame_equal(result, expected)