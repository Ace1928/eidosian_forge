import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def test_raise_exception(self):
    prob = Maratos()
    message = 'Whenever the gradient is estimated via finite-differences'
    with pytest.raises(ValueError, match=message):
        minimize(prob.fun, prob.x0, method='trust-constr', jac='2-point', hess='2-point', constraints=prob.constr)