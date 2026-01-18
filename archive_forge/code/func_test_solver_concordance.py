import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
def test_solver_concordance(self):
    f = rosen
    g = rosen_der
    h = rosen_hess
    for x0 in (self.easy_guess, self.hard_guess):
        r_dogleg = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='dogleg', options={'return_all': True})
        r_trust_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='trust-ncg', options={'return_all': True})
        r_trust_krylov = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='trust-krylov', options={'return_all': True})
        r_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='newton-cg', options={'return_all': True})
        r_iterative = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='trust-exact', options={'return_all': True})
        assert_allclose(self.x_opt, r_dogleg['x'])
        assert_allclose(self.x_opt, r_trust_ncg['x'])
        assert_allclose(self.x_opt, r_trust_krylov['x'])
        assert_allclose(self.x_opt, r_ncg['x'])
        assert_allclose(self.x_opt, r_iterative['x'])
        assert_(len(r_dogleg['allvecs']) < len(r_ncg['allvecs']))