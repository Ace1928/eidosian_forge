import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _opt_correl(self, nuis_params, corr0, endog, nobs, x0, weights0):
    """
        Parameters
        ----------
        nuis_params : 1darray
            Array containing two nuisance means and two nuisance variances

        Returns
        -------
        llr : float
            The log-likelihood of the correlation coefficient holding nuisance
            parameters constant
        """
    mu1_data, mu2_data = (endog - nuis_params[::2]).T
    sig1_data = mu1_data ** 2 - nuis_params[1]
    sig2_data = mu2_data ** 2 - nuis_params[3]
    correl_data = mu1_data * mu2_data - corr0 * (nuis_params[1] * nuis_params[3]) ** 0.5
    est_vect = np.column_stack((mu1_data, sig1_data, mu2_data, sig2_data, correl_data))
    eta_star = self._modif_newton(x0, est_vect, weights0)
    denom = 1.0 + np.dot(est_vect, eta_star)
    self.new_weights = 1.0 / nobs * 1.0 / denom
    llr = np.sum(np.log(nobs * self.new_weights))
    return -2 * llr