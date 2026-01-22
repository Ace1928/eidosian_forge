import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@ar_params.setter
def ar_params(self, value):
    if np.isscalar(value):
        value = [value] * self.k_ar_params
    self._params_split['ar_params'] = validate_basic(value, self.k_ar_params, title='AR coefficients')
    self._params = None