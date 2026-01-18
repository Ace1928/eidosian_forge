import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
def test_trust_exact_start_in_optimum(self):
    r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess, tol=1e-08, method='trust-exact')
    assert_allclose(self.x_opt, r['x'])