import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
from statsmodels.distributions import zipoisson, zigenpoisson, zinegbin
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.compat.pandas import Appender
def _get_start_params(self):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ConvergenceWarning)
        start_params = self.model_main.fit(disp=0, method='nm').params
    start_params = np.append(np.zeros(self.k_inflate), start_params)
    return start_params