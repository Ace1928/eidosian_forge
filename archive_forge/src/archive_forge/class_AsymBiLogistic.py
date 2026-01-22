import numpy as np
from scipy import stats
from statsmodels.tools.numdiff import _approx_fprime_cs_scalar, approx_hess
class AsymBiLogistic(PickandDependence):
    """bilogistic model of Coles and Tawn 1994, Joe, Smith and Weissman 1992

    restrictions:
     - (beta, delta) in (0,1)^2 or
     - (beta, delta) in (-inf,0)^2

    not vectorized because of numerical integration
    """
    k_args = 2

    def _check_args(self, beta, delta):
        cond1 = beta > 0 and beta <= 1 and (delta > 0) and (delta <= 1)
        cond2 = beta < 0 and delta < 0
        return cond1 | cond2

    def evaluate(self, t, beta, delta):

        def _integrant(w):
            term1 = (1 - beta) * np.power(w, -beta) * (1 - t)
            term2 = (1 - delta) * np.power(1 - w, -delta) * t
            return np.maximum(term1, term2)
        from scipy.integrate import quad
        transf = quad(_integrant, 0, 1)[0]
        return transf