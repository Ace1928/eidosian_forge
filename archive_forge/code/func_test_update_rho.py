import osqp
from osqp._osqp import constant
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
def test_update_rho(self):
    res_default = self.model.solve()
    default_opts = self.opts.copy()
    default_opts['rho'] = 0.7
    self.model = osqp.OSQP()
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **default_opts)
    self.model.update_settings(rho=self.opts['rho'])
    res_updated_rho = self.model.solve()
    self.assertEqual(res_default.info.iter, res_updated_rho.info.iter)