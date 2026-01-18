import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve
from numpy.testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('shape', [[2, 2], [2, 4], [4, 2], [20, 20], [20, 4], [4, 20]])
def test_simple_lu_shapes_real_complex_2d_indices(self, shape):
    a = self.rng.uniform(-10.0, 10.0, size=shape)
    p, l, u = lu(a, p_indices=True)
    assert_allclose(a, l[p, :] @ u)