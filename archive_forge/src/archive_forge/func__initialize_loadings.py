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
def _initialize_loadings(self):
    self.parameters['factor_loadings'] = self.k_endog * self.k_factors
    if self.error_order > 0:
        start = self._factor_order
        end = self._factor_order + self.k_endog
        self.ssm['design', :, start:end] = np.eye(self.k_endog)
    self._idx_loadings = np.s_['design', :, :self.k_factors]