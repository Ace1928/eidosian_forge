import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve
from numpy.testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('shape', [[2, 2], [2, 4], [4, 2], [20, 20], [20, 4], [4, 20], [3, 2, 9, 9], [2, 2, 17, 5], [2, 2, 11, 7]])
def test_simple_lu_shapes_real_complex(self, shape):
    a = self.rng.uniform(-10.0, 10.0, size=shape)
    p, l, u = lu(a)
    assert_allclose(a, p @ l @ u)
    pl, u = lu(a, permute_l=True)
    assert_allclose(a, pl @ u)
    b = self.rng.uniform(-10.0, 10.0, size=shape) * 1j
    b += self.rng.uniform(-10, 10, size=shape)
    pl, u = lu(b, permute_l=True)
    assert_allclose(b, pl @ u)