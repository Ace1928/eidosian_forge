import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _modif_newton(self, eta, est_vect, weights):
    """
        Modified Newton's method for maximizing the log 'star' equation.  This
        function calls _fit_newton to find the optimal values of eta.

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray, (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        params : 1xm array
            Lagrange multiplier that maximizes the log-likelihood
        """
    nobs = len(est_vect)
    f = lambda x0: -np.sum(self._log_star(x0, est_vect, weights, nobs))
    grad = lambda x0: -self._grad(x0, est_vect, weights, nobs)
    hess = lambda x0: -self._hess(x0, est_vect, weights, nobs)
    kwds = {'tol': 1e-08}
    eta = eta.squeeze()
    res = _fit_newton(f, grad, eta, (), kwds, hess=hess, maxiter=50, disp=0)
    return res[0]