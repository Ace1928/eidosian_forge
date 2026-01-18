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
def test_robustness(self):
    for noise in [0.1, 1.0]:
        p = ExponentialFittingProblem(1, 0.1, noise, random_seed=0)
        for jac in ['2-point', '3-point', 'cs', p.jac]:
            res_lsq = least_squares(p.fun, p.p0, jac=jac, method=self.method)
            assert_allclose(res_lsq.optimality, 0, atol=0.01)
            for loss in LOSSES:
                if loss == 'linear':
                    continue
                res_robust = least_squares(p.fun, p.p0, jac=jac, loss=loss, f_scale=noise, method=self.method)
                assert_allclose(res_robust.optimality, 0, atol=0.01)
                assert_(norm(res_robust.x - p.p_opt) < norm(res_lsq.x - p.p_opt))