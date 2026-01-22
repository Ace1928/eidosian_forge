import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
class TestApproxDerivativeSparse:

    def setup_method(self):
        np.random.seed(0)
        self.n = 50
        self.lb = -0.1 * (1 + np.arange(self.n))
        self.ub = 0.1 * (1 + np.arange(self.n))
        self.x0 = np.empty(self.n)
        self.x0[::2] = (1 - 1e-07) * self.lb[::2]
        self.x0[1::2] = (1 - 1e-07) * self.ub[1::2]
        self.J_true = self.jac(self.x0)

    def fun(self, x):
        e = x[1:] ** 3 - x[:-1] ** 2
        return np.hstack((0, 3 * e)) + np.hstack((2 * e, 0))

    def jac(self, x):
        n = x.size
        J = np.zeros((n, n))
        J[0, 0] = -4 * x[0]
        J[0, 1] = 6 * x[1] ** 2
        for i in range(1, n - 1):
            J[i, i - 1] = -6 * x[i - 1]
            J[i, i] = 9 * x[i] ** 2 - 4 * x[i]
            J[i, i + 1] = 6 * x[i + 1] ** 2
        J[-1, -1] = 9 * x[-1] ** 2
        J[-1, -2] = -6 * x[-2]
        return J

    def structure(self, n):
        A = np.zeros((n, n), dtype=int)
        A[0, 0] = 1
        A[0, 1] = 1
        for i in range(1, n - 1):
            A[i, i - 1:i + 2] = 1
        A[-1, -1] = 1
        A[-1, -2] = 1
        return A

    def test_all(self):
        A = self.structure(self.n)
        order = np.arange(self.n)
        groups_1 = group_columns(A, order)
        np.random.shuffle(order)
        groups_2 = group_columns(A, order)
        for method, groups, l, u in product(['2-point', '3-point', 'cs'], [groups_1, groups_2], [-np.inf, self.lb], [np.inf, self.ub]):
            J = approx_derivative(self.fun, self.x0, method=method, bounds=(l, u), sparsity=(A, groups))
            assert_(isinstance(J, csr_matrix))
            assert_allclose(J.toarray(), self.J_true, rtol=1e-06)
            rel_step = np.full_like(self.x0, 1e-08)
            rel_step[::2] *= -1
            J = approx_derivative(self.fun, self.x0, method=method, rel_step=rel_step, sparsity=(A, groups))
            assert_allclose(J.toarray(), self.J_true, rtol=1e-05)

    def test_no_precomputed_groups(self):
        A = self.structure(self.n)
        J = approx_derivative(self.fun, self.x0, sparsity=A)
        assert_allclose(J.toarray(), self.J_true, rtol=1e-06)

    def test_equivalence(self):
        structure = np.ones((self.n, self.n), dtype=int)
        groups = np.arange(self.n)
        for method in ['2-point', '3-point', 'cs']:
            J_dense = approx_derivative(self.fun, self.x0, method=method)
            J_sparse = approx_derivative(self.fun, self.x0, sparsity=(structure, groups), method=method)
            assert_allclose(J_dense, J_sparse.toarray(), rtol=5e-16, atol=7e-15)

    def test_check_derivative(self):

        def jac(x):
            return csr_matrix(self.jac(x))
        accuracy = check_derivative(self.fun, jac, self.x0, bounds=(self.lb, self.ub))
        assert_(accuracy < 1e-09)
        accuracy = check_derivative(self.fun, jac, self.x0, bounds=(self.lb, self.ub))
        assert_(accuracy < 1e-09)