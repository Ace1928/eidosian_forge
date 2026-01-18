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
def test_rosenbrock_bounds(self):
    x0_1 = np.array([-2.0, 1.0])
    x0_2 = np.array([2.0, 2.0])
    x0_3 = np.array([-2.0, 2.0])
    x0_4 = np.array([0.0, 2.0])
    x0_5 = np.array([-1.2, 1.0])
    problems = [(x0_1, ([-np.inf, -1.5], np.inf)), (x0_2, ([-np.inf, 1.5], np.inf)), (x0_3, ([-np.inf, 1.5], np.inf)), (x0_4, ([-np.inf, 1.5], [1.0, np.inf])), (x0_2, ([1.0, 1.5], [3.0, 3.0])), (x0_5, ([-50.0, 0.0], [0.5, 100]))]
    for x0, bounds in problems:
        for jac, x_scale, tr_solver in product(['2-point', '3-point', 'cs', jac_rosenbrock], [1.0, [1.0, 0.5], 'jac'], ['exact', 'lsmr']):
            res = least_squares(fun_rosenbrock, x0, jac, bounds, x_scale=x_scale, tr_solver=tr_solver, method=self.method)
            assert_allclose(res.optimality, 0.0, atol=1e-05)