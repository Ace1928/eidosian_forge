import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
import collections
import warnings
import itertools
def _denom(self, grp, params, ofs=None):
    if ofs is None:
        ofs = 0
    exb = np.exp(np.dot(self._exog_grp[grp], params) + ofs)
    memo = {}

    def f(t, k):
        if t < k:
            return 0
        if k == 0:
            return 1
        try:
            return memo[t, k]
        except KeyError:
            pass
        v = f(t - 1, k) + f(t - 1, k - 1) * exb[t - 1]
        memo[t, k] = v
        return v
    return f(self._groupsize[grp], self._n1[grp])