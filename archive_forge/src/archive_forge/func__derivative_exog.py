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
def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None):
    """
        For computing marginal effects. These are the marginal effects
        d F(XB) / dX
        For the Poisson model F(XB) is the predicted counts rather than
        the probabilities.

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
    if exog is None:
        exog = self.exog
    k_extra = getattr(self, 'k_extra', 0)
    params_exog = params if k_extra == 0 else params[:-k_extra]
    margeff = self.predict(params, exog)[:, None] * params_exog[None, :]
    if 'ex' in transform:
        margeff *= exog
    if 'ey' in transform:
        margeff /= self.predict(params, exog)[:, None]
    return self._derivative_exog_helper(margeff, params, exog, dummy_idx, count_idx, transform)