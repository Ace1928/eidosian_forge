from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
class CategoricalCovStruct(CovStruct):
    """
    Parent class for covariance structure for categorical data models.

    Attributes
    ----------
    nlevel : int
        The number of distinct levels for the outcome variable.
    ibd : list
        A list whose i^th element ibd[i] is an array whose rows
        contain integer pairs (a,b), where endog_li[i][a:b] is the
        subvector of binary indicators derived from the same ordinal
        value.
    """

    def initialize(self, model):
        super().initialize(model)
        self.nlevel = len(model.endog_values)
        self._ncut = self.nlevel - 1
        from numpy.lib.stride_tricks import as_strided
        b = np.dtype(np.int64).itemsize
        ibd = []
        for v in model.endog_li:
            jj = np.arange(0, len(v) + 1, self._ncut, dtype=np.int64)
            jj = as_strided(jj, shape=(len(jj) - 1, 2), strides=(b, b))
            ibd.append(jj)
        self.ibd = ibd