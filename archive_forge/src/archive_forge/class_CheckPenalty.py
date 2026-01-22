import numpy as np
from numpy.testing import assert_allclose, assert_equal
import statsmodels.base._penalties as smpen
from statsmodels.tools.numdiff import approx_fprime, approx_hess
class CheckPenalty:

    def test_symmetry(self):
        pen = self.pen
        x = self.params
        p = np.array([pen.func(np.atleast_1d(xi)) for xi in x])
        assert_allclose(p, p[::-1], rtol=1e-10)
        assert_allclose(pen.func(0 * np.atleast_1d(x[0])), 0, rtol=1e-10)

    def test_derivatives(self):
        pen = self.pen
        x = self.params
        ps = np.array([pen.deriv(np.atleast_1d(xi)) for xi in x])
        psn = np.array([approx_fprime(np.atleast_1d(xi), pen.func) for xi in x])
        assert_allclose(ps, psn, rtol=1e-07, atol=1e-08)
        ph = np.array([pen.deriv2(np.atleast_1d(xi)) for xi in x])
        phn = np.array([approx_hess(np.atleast_1d(xi), pen.func) for xi in x])
        if ph.ndim == 2:
            ph = np.array([np.diag(phi) for phi in ph])
        assert_allclose(ph, phn, rtol=1e-07, atol=1e-08)