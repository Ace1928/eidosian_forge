import osqp
from osqp._osqp import constant
from scipy import sparse
import numpy as np
import unittest
def test_primal_infeasible_problem(self):
    np.random.seed(4)
    self.n = 50
    self.m = 500
    Pt = np.random.rand(self.n, self.n)
    self.P = sparse.triu(Pt.T.dot(Pt), format='csc')
    self.q = np.random.rand(self.n)
    self.A = sparse.random(self.m, self.n).tolil()
    self.u = 3 + np.random.randn(self.m)
    self.l = -3 + np.random.randn(self.m)
    self.A[int(self.n / 2), :] = self.A[int(self.n / 2) + 1, :]
    self.l[int(self.n / 2)] = self.u[int(self.n / 2) + 1] + 10 * np.random.rand()
    self.u[int(self.n / 2)] = self.l[int(self.n / 2)] + 0.5
    self.A = self.A.tocsc()
    self.model = osqp.OSQP()
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
    res = self.model.solve()
    self.assertEqual(res.info.status_val, constant('OSQP_PRIMAL_INFEASIBLE'))