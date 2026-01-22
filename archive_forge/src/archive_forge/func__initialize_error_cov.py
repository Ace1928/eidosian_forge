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
def _initialize_error_cov(self):
    if self.error_cov_type == 'scalar':
        self._initialize_error_cov_diagonal(scalar=True)
    elif self.error_cov_type == 'diagonal':
        self._initialize_error_cov_diagonal(scalar=False)
    elif self.error_cov_type == 'unstructured':
        self._initialize_error_cov_unstructured()