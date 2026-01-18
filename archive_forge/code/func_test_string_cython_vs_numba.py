import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('agg_func', [['min', 'max'], 'min', {'B': ['min', 'max'], 'C': 'sum'}])
def test_string_cython_vs_numba(agg_func, numba_supported_reductions):
    pytest.importorskip('numba')
    agg_func, kwargs = numba_supported_reductions
    data = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    grouped = data.groupby(0)
    result = grouped.transform(agg_func, engine='numba', **kwargs)
    expected = grouped.transform(agg_func, engine='cython', **kwargs)
    tm.assert_frame_equal(result, expected)
    result = grouped[1].transform(agg_func, engine='numba', **kwargs)
    expected = grouped[1].transform(agg_func, engine='cython', **kwargs)
    tm.assert_series_equal(result, expected)