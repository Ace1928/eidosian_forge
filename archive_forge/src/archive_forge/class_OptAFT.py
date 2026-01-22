import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts
class OptAFT(_OptFuncts):
    """
    Provides optimization functions used in estimating and conducting
    inference in an AFT model.

    Methods
    ------

    _opt_wtd_nuis_regress:
        Function optimized over nuisance parameters to compute
        the profile likelihood

    _EM_test:
        Uses the modified Em algorithm of Zhou 2005 to maximize the
        likelihood of a parameter vector.
    """

    def __init__(self):
        pass

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

    def _EM_test(self, nuisance_params, params=None, param_nums=None, b0_vals=None, F=None, survidx=None, uncens_nobs=None, numcensbelow=None, km=None, uncensored=None, censored=None, maxiter=None, ftol=None):
        """
        Uses EM algorithm to compute the maximum likelihood of a test

        Parameters
        ----------

        nuisance_params : ndarray
            Vector of values to be used as nuisance params.

        maxiter : int
            Number of iterations in the EM algorithm for a parameter vector

        Returns
        -------
        -2 ''*'' log likelihood ratio at hypothesized values and
        nuisance params

        Notes
        -----
        Optional parameters are provided by the test_beta function.
        """
        iters = 0
        params[param_nums] = b0_vals
        nuis_param_index = np.int_(np.delete(np.arange(self.model.nvar), param_nums))
        params[nuis_param_index] = nuisance_params
        to_test = params.reshape(self.model.nvar, 1)
        opt_res = np.inf
        diff = np.inf
        while iters < maxiter and diff > ftol:
            F = F.flatten()
            death = np.cumsum(F[::-1])
            survivalprob = death[::-1]
            surv_point_mat = np.dot(F.reshape(-1, 1), 1.0 / survivalprob[survidx].reshape(1, -1))
            surv_point_mat = add_constant(surv_point_mat)
            summed_wts = np.cumsum(surv_point_mat, axis=1)
            wts = summed_wts[np.int_(np.arange(uncens_nobs)), numcensbelow[uncensored]]
            self.model._fit_weights = wts
            new_opt_res = self._opt_wtd_nuis_regress(to_test)
            F = self.new_weights
            diff = np.abs(new_opt_res - opt_res)
            opt_res = new_opt_res
            iters = iters + 1
        death = np.cumsum(F.flatten()[::-1])
        survivalprob = death[::-1]
        llike = -opt_res + np.sum(np.log(survivalprob[survidx]))
        wtd_km = km.flatten() / np.sum(km)
        survivalmax = np.cumsum(wtd_km[::-1])[::-1]
        llikemax = np.sum(np.log(wtd_km[uncensored])) + np.sum(np.log(survivalmax[censored]))
        if iters == maxiter:
            warnings.warn('The EM reached the maximum number of iterations', IterationLimitWarning)
        return -2 * (llike - llikemax)

    def _ci_limits_beta(self, b0, param_num=None):
        """
        Returns the difference between the log likelihood for a
        parameter and some critical value.

        Parameters
        ----------
        b0: float
            Value of a regression parameter
        param_num : int
            Parameter index of b0
        """
        return self.test_beta([b0], [param_num])[0] - self.r0