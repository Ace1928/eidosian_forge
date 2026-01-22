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
def _hessian_geom(self, params):
    exog = self.exog
    y = self.endog[:, None]
    mu = self.predict(params)[:, None]
    dim = exog.shape[1]
    hess_arr = np.empty((dim, dim))
    const_arr = mu * (1 + y) / (mu + 1) ** 2
    for i in range(dim):
        for j in range(dim):
            if j > i:
                continue
            hess_arr[i, j] = np.squeeze(np.sum(-exog[:, i, None] * exog[:, j, None] * const_arr, axis=0))
    tri_idx = np.triu_indices(dim, k=1)
    hess_arr[tri_idx] = hess_arr.T[tri_idx]
    return hess_arr