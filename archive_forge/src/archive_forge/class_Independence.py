from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
class Independence(CovStruct):
    """
    An independence working dependence structure.
    """

    @Appender(CovStruct.update.__doc__)
    def update(self, params):
        return

    @Appender(CovStruct.covariance_matrix.__doc__)
    def covariance_matrix(self, expval, index):
        dim = len(expval)
        return (np.eye(dim, dtype=np.float64), True)

    @Appender(CovStruct.covariance_matrix_solve.__doc__)
    def covariance_matrix_solve(self, expval, index, stdev, rhs):
        v = stdev ** 2
        rslt = []
        for x in rhs:
            if x.ndim == 1:
                rslt.append(x / v)
            else:
                rslt.append(x / v[:, None])
        return rslt

    def summary(self):
        return 'Observations within a cluster are modeled as being independent.'