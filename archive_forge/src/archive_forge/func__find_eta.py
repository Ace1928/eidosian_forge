import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _find_eta(self, eta):
    """
        Finding the root of sum(xi-h0)/(1+eta(xi-mu)) solves for
        eta when computing ELR for univariate mean.

        Parameters
        ----------
        eta : float
            Lagrange multiplier in the empirical likelihood maximization

        Returns
        -------
        llr : float
            n times the log likelihood value for a given value of eta
        """
    return np.sum((self.endog - self.mu0) / (1.0 + eta * (self.endog - self.mu0)))