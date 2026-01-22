from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
def _score_p(self, params):
    """
        Generalized Poisson model derivative of the log-likelihood by p-parameter

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        dldp : float
            dldp is first derivative of the loglikelihood function,
        evaluated at `p-parameter`.
        """
    if self._transparams:
        alpha = np.exp(params[-1])
    else:
        alpha = params[-1]
    params = params[:-1]
    p = self.parameterization
    y = self.endog[:, None]
    mu = self.predict(params)[:, None]
    mu_p = np.power(mu, p)
    a1 = 1 + alpha * mu_p
    a2 = mu + alpha * mu_p * y
    dp = np.sum(np.log(mu) * ((a2 - mu) * ((y - 1) / a2 - 2 / a1) + (a1 - 1) * a2 / a1 ** 2))
    return dp