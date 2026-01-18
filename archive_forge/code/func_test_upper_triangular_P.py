import osqp
from osqp._osqp import constant
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
def test_upper_triangular_P(self):
    res_default = self.model.solve()
    P_triu = sparse.triu(self.P, format='csc')
    m = osqp.OSQP()
    m.setup(P=P_triu, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
    res_triu = m.solve()
    nptest.assert_allclose(res_default.x, res_triu.x, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res_default.y, res_triu.y, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(res_default.info.obj_val, res_triu.info.obj_val, decimal=decimal_tol)