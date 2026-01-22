from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def _fix_param_names(self, params, param_names=None):
    xnames = self.data.xnames
    if param_names is not None:
        if len(params) == len(param_names):
            self.data.xnames = param_names
        else:
            raise ValueError('param_names has the wrong length')
    elif len(params) < len(xnames):
        self.data.xnames = xnames[-len(params):]
    elif len(params) > len(xnames):
        self.data.xnames = ['p%2d' % i for i in range(len(params))]