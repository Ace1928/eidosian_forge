from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import raises as assert_raises
from numpy import array, transpose, dot, conjugate, zeros_like, empty
from numpy.random import random
from scipy.linalg import cholesky, cholesky_banded, cho_solve_banded, \
from scipy.linalg._testutils import assert_no_overwrite
def test_upper_real(self):
    a = array([[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.5, 0.0], [0.0, 0.5, 4.0, 0.2], [0.0, 0.0, 0.2, 4.0]])
    ab = array([[-1.0, 1.0, 0.5, 0.2], [4.0, 4.0, 4.0, 4.0]])
    c = cholesky_banded(ab, lower=False)
    ufac = zeros_like(a)
    ufac[list(range(4)), list(range(4))] = c[-1]
    ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
    assert_array_almost_equal(a, dot(ufac.T, ufac))
    b = array([0.0, 0.5, 4.2, 4.2])
    x = cho_solve_banded((c, False), b)
    assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])