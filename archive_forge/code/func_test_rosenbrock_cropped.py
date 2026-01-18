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
def test_rosenbrock_cropped(self):
    x0 = [-2, 1]
    if self.method == 'lm':
        assert_raises(ValueError, least_squares, fun_rosenbrock_cropped, x0, method='lm')
    else:
        for jac, x_scale, tr_solver in product(['2-point', '3-point', 'cs', jac_rosenbrock_cropped], [1.0, np.array([1.0, 0.2]), 'jac'], ['exact', 'lsmr']):
            res = least_squares(fun_rosenbrock_cropped, x0, jac, x_scale=x_scale, tr_solver=tr_solver, method=self.method)
            assert_allclose(res.cost, 0, atol=1e-14)