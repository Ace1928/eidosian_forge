from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def update_grid(self, params):
    endog = self.model.endog_li
    cached_means = self.model.cached_means
    varfunc = self.model.family.variance
    dep_params = np.zeros(self.max_lag + 1)
    for i in range(self.model.num_group):
        expval, _ = cached_means[i]
        stdev = np.sqrt(varfunc(expval))
        resid = (endog[i] - expval) / stdev
        dep_params[0] += np.sum(resid * resid) / len(resid)
        for j in range(1, self.max_lag + 1):
            v = resid[j:]
            dep_params[j] += np.sum(resid[0:-j] * v) / len(v)
    dep_params /= dep_params[0]
    self.dep_params = dep_params