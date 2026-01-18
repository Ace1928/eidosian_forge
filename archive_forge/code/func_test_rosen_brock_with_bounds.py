import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
@pytest.mark.parametrize('bounds, x_opt', [(Bounds(-np.inf, np.inf), Rosenbrock().x_opt), (Bounds(-np.inf, -0.8), [-0.8, -0.8]), (Bounds(3.0, np.inf), [3.0, 9.0]), (Bounds([3.0, 1.0], [4.0, 5.0]), [3.0, 5.0])])
def test_rosen_brock_with_bounds(self, bounds, x_opt):
    prob = Rosenbrock()
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'Initial guess is not within the specified bounds')
        result = minimize(prob.fun, [-10, -10], method='Nelder-Mead', bounds=bounds)
        assert np.less_equal(bounds.lb, result.x).all()
        assert np.less_equal(result.x, bounds.ub).all()
        assert np.allclose(prob.fun(result.x), result.fun)
        assert np.allclose(result.x, x_opt, atol=0.001)