import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve
from numpy.testing import assert_allclose, assert_array_equal
class TestLUFactor:

    def setup_method(self):
        self.rng = np.random.default_rng(1682281250228846)
        self.a = np.array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
        self.ca = np.array([[1, 2, 3], [1, 2, 3], [2, 5j, 6]])
        self.b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.cb = np.array([[1j, 2j, 3j], [4j, 5j, 6j], [7j, 8j, 9j]])
        self.hrect = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 12]])
        self.chrect = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 12]]) * 1j
        self.vrect = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 12, 12]])
        self.cvrect = 1j * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 12, 12]])
        self.med = self.rng.random((30, 40))
        self.cmed = self.rng.random((30, 40)) + 1j * self.rng.random((30, 40))

    def _test_common_lu_factor(self, data):
        l_and_u1, piv1 = lu_factor(data)
        getrf, = get_lapack_funcs(('getrf',), (data,))
        l_and_u2, piv2, _ = getrf(data, overwrite_a=False)
        assert_allclose(l_and_u1, l_and_u2)
        assert_allclose(piv1, piv2)

    def test_hrectangular(self):
        self._test_common_lu_factor(self.hrect)

    def test_vrectangular(self):
        self._test_common_lu_factor(self.vrect)

    def test_hrectangular_complex(self):
        self._test_common_lu_factor(self.chrect)

    def test_vrectangular_complex(self):
        self._test_common_lu_factor(self.cvrect)

    def test_medium1(self):
        """Check lu decomposition on medium size, rectangular matrix."""
        self._test_common_lu_factor(self.med)

    def test_medium1_complex(self):
        """Check lu decomposition on medium size, rectangular matrix."""
        self._test_common_lu_factor(self.cmed)

    def test_check_finite(self):
        p, l, u = lu(self.a, check_finite=False)
        assert_allclose(p @ l @ u, self.a)

    def test_simple_known(self):
        for order in ['C', 'F']:
            A = np.array([[2, 1], [0, 1.0]], order=order)
            LU, P = lu_factor(A)
            assert_allclose(LU, np.array([[2, 1], [0, 1]]))
            assert_array_equal(P, np.array([0, 1]))