import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve
from numpy.testing import assert_allclose, assert_array_equal
class TestLUSolve:

    def setup_method(self):
        self.rng = np.random.default_rng(1682281250228846)

    def test_lu(self):
        a0 = self.rng.random((10, 10))
        b = self.rng.random((10,))
        for order in ['C', 'F']:
            a = np.array(a0, order=order)
            x1 = solve(a, b)
            lu_a = lu_factor(a)
            x2 = lu_solve(lu_a, b)
            assert_allclose(x1, x2)

    def test_check_finite(self):
        a = self.rng.random((10, 10))
        b = self.rng.random((10,))
        x1 = solve(a, b)
        lu_a = lu_factor(a, check_finite=False)
        x2 = lu_solve(lu_a, b, check_finite=False)
        assert_allclose(x1, x2)