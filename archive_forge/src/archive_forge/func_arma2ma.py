from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
@Appender(remove_parameters(arma2ma.__doc__, ['ar', 'ma']))
def arma2ma(self, lags=None):
    lags = lags or self.lags
    return arma2ma(self.ar, self.ma, lags=lags)