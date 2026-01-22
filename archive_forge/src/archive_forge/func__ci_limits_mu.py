import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _ci_limits_mu(self, mu):
    """
        Calculates the difference between the log likelihood of mu_test and a
        specified critical value.

        Parameters
        ----------
        mu : float
           Hypothesized value of the mean.

        Returns
        -------
        diff : float
            The difference between the log likelihood value of mu0 and
            a specified value.
        """
    return self.test_mean(mu)[0] - self.r0