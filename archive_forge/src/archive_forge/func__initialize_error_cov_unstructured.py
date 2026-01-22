import numpy as np
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
from statsmodels.multivariate.pca import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.compat.pandas import Appender
def _initialize_error_cov_unstructured(self):
    k_endog = self.k_endog
    self.parameters['error_cov'] = int(k_endog * (k_endog + 1) / 2)
    self._idx_lower_error_cov = np.tril_indices(self.k_endog)
    if self.error_order > 0:
        start = self.k_factors
        end = self.k_factors + self.k_endog
        self._idx_error_cov = np.s_['state_cov', start:end, start:end]
    else:
        self._idx_error_cov = np.s_['obs_cov', :, :]