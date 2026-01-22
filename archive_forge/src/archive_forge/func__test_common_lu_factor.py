import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve
from numpy.testing import assert_allclose, assert_array_equal
def _test_common_lu_factor(self, data):
    l_and_u1, piv1 = lu_factor(data)
    getrf, = get_lapack_funcs(('getrf',), (data,))
    l_and_u2, piv2, _ = getrf(data, overwrite_a=False)
    assert_allclose(l_and_u1, l_and_u2)
    assert_allclose(piv1, piv2)