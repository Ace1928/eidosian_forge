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
def _initialize_exog(self):
    self.parameters['exog'] = self.k_exog * self.k_endog
    if self.k_exog > 0:
        self.ssm['obs_intercept'] = np.zeros((self.k_endog, self.nobs))
    self._idx_exog = np.s_['obs_intercept', :self.k_endog, :]