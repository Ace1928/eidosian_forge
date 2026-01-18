from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def update_nogrid(self, params):
    endog = self.model.endog_li
    cached_means = self.model.cached_means
    varfunc = self.model.family.variance
    dep_params = np.zeros(self.max_lag + 1)
    dn = np.zeros(self.max_lag + 1)
    resid_ssq = 0
    resid_ssq_n = 0
    for i in range(self.model.num_group):
        expval, _ = cached_means[i]
        stdev = np.sqrt(varfunc(expval))
        resid = (endog[i] - expval) / stdev
        j1, j2 = np.tril_indices(len(expval), -1)
        dx = np.abs(self.time[i][j1] - self.time[i][j2])
        ii = np.flatnonzero(dx <= self.max_lag)
        j1 = j1[ii]
        j2 = j2[ii]
        dx = dx[ii]
        vs = np.bincount(dx, weights=resid[j1] * resid[j2], minlength=self.max_lag + 1)
        vd = np.bincount(dx, minlength=self.max_lag + 1)
        resid_ssq += np.sum(resid ** 2)
        resid_ssq_n += len(resid)
        ii = np.flatnonzero(vd > 0)
        if len(ii) > 0:
            dn[ii] += 1
            dep_params[ii] += vs[ii] / vd[ii]
    i0 = np.flatnonzero(dn > 0)
    dep_params[i0] /= dn[i0]
    resid_msq = resid_ssq / resid_ssq_n
    dep_params /= resid_msq
    self.dep_params = dep_params