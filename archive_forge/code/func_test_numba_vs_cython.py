import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False])
@pytest.mark.parametrize('pandas_obj', ['Series', 'DataFrame'])
@pytest.mark.parametrize('as_index', [True, False])
def test_numba_vs_cython(jit, pandas_obj, nogil, parallel, nopython, as_index):
    pytest.importorskip('numba')

    def func(values, index):
        return values + 1
    if jit:
        import numba
        func = numba.jit(func)
    data = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    grouped = data.groupby(0, as_index=as_index)
    if pandas_obj == 'Series':
        grouped = grouped[1]
    result = grouped.transform(func, engine='numba', engine_kwargs=engine_kwargs)
    expected = grouped.transform(lambda x: x + 1, engine='cython')
    tm.assert_equal(result, expected)