import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [DataFrame(np.eye(5)), DataFrame([[5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3], [5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3], [np.nan, np.nan, 5, 6, 7, 5, 5, 5, 5, 5]]).T, Series(range(5), name='foo'), Series([20, 10, 10, np.inf, 1, 1, 2, 3]), Series([20, 10, 10, np.nan, 10, 1, 2, 3])])
def test_numba_vs_cython_rolling_methods(self, data, nogil, parallel, nopython, arithmetic_numba_supported_operators, step):
    method, kwargs = arithmetic_numba_supported_operators
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    roll = data.rolling(3, step=step)
    result = getattr(roll, method)(engine='numba', engine_kwargs=engine_kwargs, **kwargs)
    expected = getattr(roll, method)(engine='cython', **kwargs)
    tm.assert_equal(result, expected)