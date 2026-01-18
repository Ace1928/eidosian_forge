from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector
def test_with_bounds(self):
    p = BroydenTridiagonal()
    for jac, jac_sparsity in product([p.jac, '2-point', '3-point', 'cs'], [None, p.sparsity]):
        res_1 = least_squares(p.fun, p.x0, jac, bounds=(p.lb, np.inf), method=self.method, jac_sparsity=jac_sparsity)
        res_2 = least_squares(p.fun, p.x0, jac, bounds=(-np.inf, p.ub), method=self.method, jac_sparsity=jac_sparsity)
        res_3 = least_squares(p.fun, p.x0, jac, bounds=(p.lb, p.ub), method=self.method, jac_sparsity=jac_sparsity)
        assert_allclose(res_1.optimality, 0, atol=1e-10)
        assert_allclose(res_2.optimality, 0, atol=1e-10)
        assert_allclose(res_3.optimality, 0, atol=1e-10)