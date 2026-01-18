import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [np.eye(3), np.ones((2, 3)), np.ones((3, 2))])
@pytest.mark.parametrize('method', ['mean', 'sum'])
def test_table_method_ewm(self, data, method, axis, nogil, parallel, nopython):
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    df = DataFrame(data)
    result = getattr(df.ewm(com=1, method='table', axis=axis), method)(engine_kwargs=engine_kwargs, engine='numba')
    expected = getattr(df.ewm(com=1, method='single', axis=axis), method)(engine_kwargs=engine_kwargs, engine='numba')
    tm.assert_frame_equal(result, expected)