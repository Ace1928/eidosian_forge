from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
class Autoregressive(CovStruct):
    """
    A first-order autoregressive working dependence structure.

    The dependence is defined in terms of the `time` component of the
    parent GEE class, which defaults to the index position of each
    value within its cluster, based on the order of values in the
    input data set.  Time represents a potentially multidimensional
    index from which distances between pairs of observations can be
    determined.

    The correlation between two observations in the same cluster is
    dep_params^distance, where `dep_params` contains the (scalar)
    autocorrelation parameter to be estimated, and `distance` is the
    distance between the two observations, calculated from their
    corresponding time values.  `time` is stored as an n_obs x k
    matrix, where `k` represents the number of dimensions in the time
    index.

    The autocorrelation parameter is estimated using weighted
    nonlinear least squares, regressing each value within a cluster on
    each preceding value in the same cluster.

    Parameters
    ----------
    dist_func : function from R^k x R^k to R^+, optional
        A function that computes the distance between the two
        observations based on their `time` values.

    References
    ----------
    B Rosner, A Munoz.  Autoregressive modeling for the analysis of
    longitudinal data with unequally spaced examinations.  Statistics
    in medicine. Vol 7, 59-71, 1988.
    """

    def __init__(self, dist_func=None, grid=None):
        super().__init__()
        grid = bool_like(grid, 'grid', optional=True)
        if dist_func is None:
            self.dist_func = lambda x, y: np.abs(x - y).sum()
        else:
            self.dist_func = dist_func
        if grid is None:
            warnings.warn('grid=True will become default in a future version', FutureWarning)
        self.grid = bool(grid)
        if not self.grid:
            self.designx = None
        self.dep_params = 0.0

    @Appender(CovStruct.update.__doc__)
    def update(self, params):
        if self.model.weights is not None:
            warnings.warn('weights not implemented for autoregressive cov_struct, using unweighted covariance estimate', NotImplementedWarning)
        if self.grid:
            self._update_grid(params)
        else:
            self._update_nogrid(params)

    def _update_grid(self, params):
        cached_means = self.model.cached_means
        scale = self.model.estimate_scale()
        varfunc = self.model.family.variance
        endog = self.model.endog_li
        lag0, lag1 = (0.0, 0.0)
        for i in range(self.model.num_group):
            expval, _ = cached_means[i]
            stdev = np.sqrt(scale * varfunc(expval))
            resid = (endog[i] - expval) / stdev
            n = len(resid)
            if n > 1:
                lag1 += np.sum(resid[0:-1] * resid[1:]) / (n - 1)
                lag0 += np.sum(resid ** 2) / n
        self.dep_params = lag1 / lag0

    def _update_nogrid(self, params):
        endog = self.model.endog_li
        time = self.model.time_li
        if self.designx is not None:
            designx = self.designx
        else:
            designx = []
            for i in range(self.model.num_group):
                ngrp = len(endog[i])
                if ngrp == 0:
                    continue
                for j1 in range(ngrp):
                    for j2 in range(j1):
                        designx.append(self.dist_func(time[i][j1, :], time[i][j2, :]))
            designx = np.array(designx)
            self.designx = designx
        scale = self.model.estimate_scale()
        varfunc = self.model.family.variance
        cached_means = self.model.cached_means
        var = 1.0 - self.dep_params ** (2 * designx)
        var /= 1.0 - self.dep_params ** 2
        wts = 1.0 / var
        wts /= wts.sum()
        residmat = []
        for i in range(self.model.num_group):
            expval, _ = cached_means[i]
            stdev = np.sqrt(scale * varfunc(expval))
            resid = (endog[i] - expval) / stdev
            ngrp = len(resid)
            for j1 in range(ngrp):
                for j2 in range(j1):
                    residmat.append([resid[j1], resid[j2]])
        residmat = np.array(residmat)

        def fitfunc(a):
            dif = residmat[:, 0] - a ** designx * residmat[:, 1]
            return np.dot(dif ** 2, wts)
        b_lft, f_lft = (0.0, fitfunc(0.0))
        b_ctr, f_ctr = (0.5, fitfunc(0.5))
        while f_ctr > f_lft:
            b_ctr /= 2
            f_ctr = fitfunc(b_ctr)
            if b_ctr < 1e-08:
                self.dep_params = 0
                return
        b_rgt, f_rgt = (0.75, fitfunc(0.75))
        while f_rgt < f_ctr:
            b_rgt = b_rgt + (1.0 - b_rgt) / 2
            f_rgt = fitfunc(b_rgt)
            if b_rgt > 1.0 - 1e-06:
                raise ValueError('Autoregressive: unable to find right bracket')
        from scipy.optimize import brent
        self.dep_params = brent(fitfunc, brack=[b_lft, b_ctr, b_rgt])

    @Appender(CovStruct.covariance_matrix.__doc__)
    def covariance_matrix(self, endog_expval, index):
        ngrp = len(endog_expval)
        if self.dep_params == 0:
            return (np.eye(ngrp, dtype=np.float64), True)
        idx = np.arange(ngrp)
        cmat = self.dep_params ** np.abs(idx[:, None] - idx[None, :])
        return (cmat, True)

    @Appender(CovStruct.covariance_matrix_solve.__doc__)
    def covariance_matrix_solve(self, expval, index, stdev, rhs):
        k = len(expval)
        r = self.dep_params
        soln = []
        if k == 1:
            return [x / stdev ** 2 for x in rhs]
        if k == 2:
            mat = np.array([[1, -r], [-r, 1]])
            mat /= 1.0 - r ** 2
            for x in rhs:
                if x.ndim == 1:
                    x1 = x / stdev
                else:
                    x1 = x / stdev[:, None]
                x1 = np.dot(mat, x1)
                if x.ndim == 1:
                    x1 /= stdev
                else:
                    x1 /= stdev[:, None]
                soln.append(x1)
            return soln
        c0 = (1.0 + r ** 2) / (1.0 - r ** 2)
        c1 = 1.0 / (1.0 - r ** 2)
        c2 = -r / (1.0 - r ** 2)
        soln = []
        for x in rhs:
            flatten = False
            if x.ndim == 1:
                x = x[:, None]
                flatten = True
            x1 = x / stdev[:, None]
            z0 = np.zeros((1, x1.shape[1]))
            rhs1 = np.concatenate((x1[1:, :], z0), axis=0)
            rhs2 = np.concatenate((z0, x1[0:-1, :]), axis=0)
            y = c0 * x1 + c2 * rhs1 + c2 * rhs2
            y[0, :] = c1 * x1[0, :] + c2 * x1[1, :]
            y[-1, :] = c1 * x1[-1, :] + c2 * x1[-2, :]
            y /= stdev[:, None]
            if flatten:
                y = np.squeeze(y)
            soln.append(y)
        return soln

    def summary(self):
        return 'Autoregressive(1) dependence parameter: %.3f\n' % self.dep_params