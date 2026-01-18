import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [DataFrame(np.eye(5)), Series(range(5), name='foo')])
def test_numba_vs_cython_expanding_methods(self, data, nogil, parallel, nopython, arithmetic_numba_supported_operators):
    method, kwargs = arithmetic_numba_supported_operators
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    data = DataFrame(np.eye(5))
    expand = data.expanding()
    result = getattr(expand, method)(engine='numba', engine_kwargs=engine_kwargs, **kwargs)
    expected = getattr(expand, method)(engine='cython', **kwargs)
    tm.assert_equal(result, expected)