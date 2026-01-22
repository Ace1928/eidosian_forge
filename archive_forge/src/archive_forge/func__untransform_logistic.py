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
def _untransform_logistic(self, unconstrained, constrained):
    """
        Function to allow using a numerical root-finder to reverse the
        logistic transform.
        """
    resid = np.zeros(unconstrained.shape, dtype=unconstrained.dtype)
    exp = np.exp(unconstrained)
    sum_exp = np.sum(exp)
    for i in range(len(unconstrained)):
        resid[i] = unconstrained[i] - np.log(1 + sum_exp - exp[i]) + np.log(1 / constrained[i] - 1)
    return resid