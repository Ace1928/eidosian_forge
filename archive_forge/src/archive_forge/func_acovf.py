from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
@Appender(remove_parameters(arma_acovf.__doc__, ['ar', 'ma', 'sigma2']))
def acovf(self, nobs=None):
    nobs = nobs or self.nobs
    return arma_acovf(self.ar, self.ma, nobs=nobs)