import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
def _regime_transition_matrix_tvtp(self, params, exog_tvtp=None):
    if exog_tvtp is None:
        exog_tvtp = self.exog_tvtp
    nobs = len(exog_tvtp)
    regime_transition_matrix = np.zeros((self.k_regimes, self.k_regimes, nobs), dtype=np.promote_types(np.float64, params.dtype))
    for i in range(self.k_regimes):
        coeffs = params[self.parameters[i, 'regime_transition']]
        regime_transition_matrix[:-1, i, :] = np.dot(exog_tvtp, np.reshape(coeffs, (self.k_regimes - 1, self.k_tvtp)).T).T
    tmp = np.c_[np.zeros((nobs, self.k_regimes, 1)), regime_transition_matrix[:-1, :, :].T].T
    regime_transition_matrix[:-1, :, :] = np.exp(regime_transition_matrix[:-1, :, :] - logsumexp(tmp, axis=0))
    regime_transition_matrix[-1, :, :] = 1 - np.sum(regime_transition_matrix[:-1, :, :], axis=0)
    return regime_transition_matrix