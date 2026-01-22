import osqp
import numpy as np
from scipy import sparse
import unittest
class warm_start_tests(unittest.TestCase):

    def setUp(self):
        """
        Setup default options
        """
        self.opts = {'verbose': False, 'adaptive_rho': False, 'eps_abs': 1e-08, 'eps_rel': 1e-08, 'polish': False, 'check_termination': 1}

    def test_warm_start(self):
        np.random.seed(2)
        self.n = 100
        self.m = 200
        self.A = sparse.random(self.m, self.n, density=0.9, format='csc')
        self.l = -np.random.rand(self.m) * 2.0
        self.u = np.random.rand(self.m) * 2.0
        P = sparse.random(self.n, self.n, density=0.9)
        self.P = sparse.triu(P.dot(P.T), format='csc')
        self.q = np.random.randn(self.n)
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
        res = self.model.solve()
        x_opt = res.x
        y_opt = res.y
        tot_iter = res.info.iter
        self.model.warm_start(x=np.zeros(self.n), y=np.zeros(self.m))
        res = self.model.solve()
        self.assertEqual(res.info.iter, tot_iter)
        self.model.warm_start(x=x_opt, y=y_opt)
        res = self.model.solve()
        self.assertLess(res.info.iter, 10)