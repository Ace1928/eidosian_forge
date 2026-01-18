from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
@cache_readonly
def var_rep(self):
    pi = self.alpha.dot(self.beta.T)
    gamma = self.gamma
    K = self.neqs
    A = np.zeros((self.k_ar, K, K))
    A[0] = pi + np.identity(K)
    if self.gamma.size > 0:
        A[0] += gamma[:, :K]
        A[self.k_ar - 1] = -gamma[:, K * (self.k_ar - 2):]
        for i in range(1, self.k_ar - 1):
            A[i] = gamma[:, K * i:K * (i + 1)] - gamma[:, K * (i - 1):K * i]
    return A