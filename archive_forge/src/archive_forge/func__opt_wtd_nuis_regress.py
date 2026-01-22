import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts
def _opt_wtd_nuis_regress(self, test_vals):
    """
        A function that is optimized over nuisance parameters to conduct a
        hypothesis test for the parameters of interest

        Parameters
        ----------

        params: 1d array
            The regression coefficients of the model.  This includes the
            nuisance and parameters of interests.

        Returns
        -------
        llr : float
            -2 times the log likelihood of the nuisance parameters and the
            hypothesized value of the parameter(s) of interest.
        """
    test_params = test_vals.reshape(self.model.nvar, 1)
    est_vect = self.model.uncens_exog * (self.model.uncens_endog - np.dot(self.model.uncens_exog, test_params))
    eta_star = self._modif_newton(np.zeros(self.model.nvar), est_vect, self.model._fit_weights)
    denom = np.sum(self.model._fit_weights) + np.dot(eta_star, est_vect.T)
    self.new_weights = self.model._fit_weights / denom
    return -1 * np.sum(np.log(self.new_weights))