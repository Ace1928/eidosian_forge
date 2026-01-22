from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
class Stationary(CovStruct):
    """
    A stationary covariance structure.

    The correlation between two observations is an arbitrary function
    of the distance between them.  Distances up to a given maximum
    value are included in the covariance model.

    Parameters
    ----------
    max_lag : float
        The largest distance that is included in the covariance model.
    grid : bool
        If True, the index positions in the data (after dropping missing
        values) are used to define distances, and the `time` variable is
        ignored.
    """

    def __init__(self, max_lag=1, grid=None):
        super().__init__()
        grid = bool_like(grid, 'grid', optional=True)
        if grid is None:
            warnings.warn('grid=True will become default in a future version', FutureWarning)
        self.max_lag = max_lag
        self.grid = bool(grid)
        self.dep_params = np.zeros(max_lag + 1)

    def initialize(self, model):
        super().initialize(model)
        if not self.grid:
            time = self.model.time[:, 0].astype(np.int32)
            self.time = self.model.cluster_list(time)

    @Appender(CovStruct.update.__doc__)
    def update(self, params):
        if self.grid:
            self.update_grid(params)
        else:
            self.update_nogrid(params)

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

    @Appender(CovStruct.covariance_matrix.__doc__)
    def covariance_matrix(self, endog_expval, index):
        if self.grid:
            return self.covariance_matrix_grid(endog_expval, index)
        j1, j2 = np.tril_indices(len(endog_expval), -1)
        dx = np.abs(self.time[index][j1] - self.time[index][j2])
        ii = np.flatnonzero(dx <= self.max_lag)
        j1 = j1[ii]
        j2 = j2[ii]
        dx = dx[ii]
        cmat = np.eye(len(endog_expval))
        cmat[j1, j2] = self.dep_params[dx]
        cmat[j2, j1] = self.dep_params[dx]
        return (cmat, True)

    def covariance_matrix_grid(self, endog_expval, index):
        from scipy.linalg import toeplitz
        r = np.zeros(len(endog_expval))
        r[0] = 1
        r[1:self.max_lag + 1] = self.dep_params[1:]
        return (toeplitz(r), True)

    @Appender(CovStruct.covariance_matrix_solve.__doc__)
    def covariance_matrix_solve(self, expval, index, stdev, rhs):
        if not self.grid:
            return super().covariance_matrix_solve(expval, index, stdev, rhs)
        from statsmodels.tools.linalg import stationary_solve
        r = np.zeros(len(expval))
        r[0:self.max_lag] = self.dep_params[1:]
        rslt = []
        for x in rhs:
            if x.ndim == 1:
                y = x / stdev
                rslt.append(stationary_solve(r, y) / stdev)
            else:
                y = x / stdev[:, None]
                rslt.append(stationary_solve(r, y) / stdev[:, None])
        return rslt

    def summary(self):
        lag = np.arange(self.max_lag + 1)
        return pd.DataFrame({'Lag': lag, 'Cov': self.dep_params})