import osqp
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
def test_update_A(self):
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()
    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_A')
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=decimal_tol)