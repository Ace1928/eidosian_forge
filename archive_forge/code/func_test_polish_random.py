import osqp
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
def test_polish_random(self):
    np.random.seed(6)
    self.n = 30
    self.m = 50
    Pt = sparse.random(self.n, self.n)
    self.P = Pt.T @ Pt
    self.q = np.random.randn(self.n)
    self.A = sparse.csc_matrix(np.random.randn(self.m, self.n))
    self.l = -3 + np.random.randn(self.m)
    self.u = 3 + np.random.randn(self.m)
    self.model = osqp.OSQP()
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
    res = self.model.solve()
    x_sol, y_sol, obj_sol = load_high_accuracy('test_polish_random')
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=decimal_tol)