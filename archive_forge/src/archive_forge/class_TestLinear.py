from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
class TestLinear:
    """Solve a linear equation;
    some methods find the exact solution in a finite number of steps"""

    def _check(self, jac, N, maxiter, complex=False, **kw):
        np.random.seed(123)
        A = np.random.randn(N, N)
        if complex:
            A = A + 1j * np.random.randn(N, N)
        b = np.random.randn(N)
        if complex:
            b = b + 1j * np.random.randn(N)

        def func(x):
            return dot(A, x) - b
        sol = nonlin.nonlin_solve(func, np.zeros(N), jac, maxiter=maxiter, f_tol=1e-06, line_search=None, verbose=0)
        assert_(np.allclose(dot(A, sol), b, atol=1e-06))

    def test_broyden1(self):
        self._check(nonlin.BroydenFirst(alpha=1.0), 20, 41, False)
        self._check(nonlin.BroydenFirst(alpha=1.0), 20, 41, True)

    def test_broyden2(self):
        self._check(nonlin.BroydenSecond(alpha=1.0), 20, 41, False)
        self._check(nonlin.BroydenSecond(alpha=1.0), 20, 41, True)

    def test_anderson(self):
        self._check(nonlin.Anderson(M=50, alpha=1.0), 20, 29, False)
        self._check(nonlin.Anderson(M=50, alpha=1.0), 20, 29, True)

    def test_krylov(self):
        self._check(nonlin.KrylovJacobian, 20, 2, False, inner_m=10)
        self._check(nonlin.KrylovJacobian, 20, 2, True, inner_m=10)

    def _check_autojac(self, A, b):

        def func(x):
            return A.dot(x) - b

        def jac(v):
            return A
        sol = nonlin.nonlin_solve(func, np.zeros(b.shape[0]), jac, maxiter=2, f_tol=1e-06, line_search=None, verbose=0)
        np.testing.assert_allclose(A @ sol, b, atol=1e-06)
        sol = nonlin.nonlin_solve(func, np.zeros(b.shape[0]), A, maxiter=2, f_tol=1e-06, line_search=None, verbose=0)
        np.testing.assert_allclose(A @ sol, b, atol=1e-06)

    def test_jac_sparse(self):
        A = csr_array([[1, 2], [2, 1]])
        b = np.array([1, -1])
        self._check_autojac(A, b)
        self._check_autojac((1 + 2j) * A, (2 + 2j) * b)

    def test_jac_ndarray(self):
        A = np.array([[1, 2], [2, 1]])
        b = np.array([1, -1])
        self._check_autojac(A, b)
        self._check_autojac((1 + 2j) * A, (2 + 2j) * b)