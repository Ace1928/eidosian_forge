import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def _llobs(self, endog, exog, exog_precision, params):
    """
        Loglikelihood for observations with data arguments.

        Parameters
        ----------
        endog : ndarray
            1d array of endogenous variable.
        exog : ndarray
            2d array of explanatory variables.
        exog_precision : ndarray
            2d array of explanatory variables for precision.
        params : ndarray
            The parameters of the model, coefficients for linear predictors
            of the mean and of the precision function.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`.
        """
    y, X, Z = (endog, exog, exog_precision)
    nz = Z.shape[1]
    params_mean = params[:-nz]
    params_prec = params[-nz:]
    linpred = np.dot(X, params_mean)
    linpred_prec = np.dot(Z, params_prec)
    mu = self.link.inverse(linpred)
    phi = self.link_precision.inverse(linpred_prec)
    eps_lb = 1e-200
    alpha = np.clip(mu * phi, eps_lb, np.inf)
    beta = np.clip((1 - mu) * phi, eps_lb, np.inf)
    ll = lgamma(phi) - lgamma(alpha) - lgamma(beta) + (mu * phi - 1) * np.log(y) + ((1 - mu) * phi - 1) * np.log(1 - y)
    return ll