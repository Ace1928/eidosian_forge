from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
class Exchangeable(CovStruct):
    """
    An exchangeable working dependence structure.
    """

    def __init__(self):
        super().__init__()
        self.dep_params = 0.0

    @Appender(CovStruct.update.__doc__)
    def update(self, params):
        endog = self.model.endog_li
        nobs = self.model.nobs
        varfunc = self.model.family.variance
        cached_means = self.model.cached_means
        has_weights = self.model.weights is not None
        weights_li = self.model.weights
        residsq_sum, scale = (0, 0)
        fsum1, fsum2, n_pairs = (0.0, 0.0, 0.0)
        for i in range(self.model.num_group):
            expval, _ = cached_means[i]
            stdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / stdev
            f = weights_li[i] if has_weights else 1.0
            ssr = np.sum(resid * resid)
            scale += f * ssr
            fsum1 += f * len(endog[i])
            residsq_sum += f * (resid.sum() ** 2 - ssr) / 2
            ngrp = len(resid)
            npr = 0.5 * ngrp * (ngrp - 1)
            fsum2 += f * npr
            n_pairs += npr
        ddof = self.model.ddof_scale
        scale /= fsum1 * (nobs - ddof) / float(nobs)
        residsq_sum /= scale
        self.dep_params = residsq_sum / (fsum2 * (n_pairs - ddof) / float(n_pairs))

    @Appender(CovStruct.covariance_matrix.__doc__)
    def covariance_matrix(self, expval, index):
        dim = len(expval)
        dp = self.dep_params * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(dp, 1)
        return (dp, True)

    @Appender(CovStruct.covariance_matrix_solve.__doc__)
    def covariance_matrix_solve(self, expval, index, stdev, rhs):
        k = len(expval)
        c = self.dep_params / (1.0 - self.dep_params)
        c /= 1.0 + self.dep_params * (k - 1)
        rslt = []
        for x in rhs:
            if x.ndim == 1:
                x1 = x / stdev
                y = x1 / (1.0 - self.dep_params)
                y -= c * sum(x1)
                y /= stdev
            else:
                x1 = x / stdev[:, None]
                y = x1 / (1.0 - self.dep_params)
                y -= c * x1.sum(0)
                y /= stdev[:, None]
            rslt.append(y)
        return rslt

    def summary(self):
        return 'The correlation between two observations in the ' + 'same cluster is %.3f' % self.dep_params