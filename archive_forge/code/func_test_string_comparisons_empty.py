import pytest
import operator
import numpy as np
from numpy.testing import assert_array_equal
@pytest.mark.parametrize(['op', 'ufunc', 'sym'], COMPARISONS)
@pytest.mark.parametrize('dtypes', [('S2', 'S2'), ('S2', 'S10'), ('<U1', '<U1'), ('<U1', '>U10')])
def test_string_comparisons_empty(op, ufunc, sym, dtypes):
    arr = np.empty((1, 0, 1, 5), dtype=dtypes[0])
    arr2 = np.empty((100, 1, 0, 1), dtype=dtypes[1])
    expected = np.empty(np.broadcast_shapes(arr.shape, arr2.shape), dtype=bool)
    assert_array_equal(op(arr, arr2), expected)
    assert_array_equal(ufunc(arr, arr2), expected)
    assert_array_equal(np.compare_chararrays(arr, arr2, sym, False), expected)