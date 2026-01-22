import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def _compute_lambda(self, Y, X):
    """Computes only lambda -- the main part of the test statistic"""
    n = np.shape(X)[0]
    Y = _adjust_shape(Y, 1)
    X = _adjust_shape(X, self.k_vars)
    b = KernelReg(Y, X, self.var_type, self.model.reg_type, self.bw, defaults=EstimatorSettings(efficient=False)).fit()[1]
    b = b[:, self.test_vars]
    b = np.reshape(b, (n, len(self.test_vars)))
    fct = 1.0
    lam = ((b / fct) ** 2).sum() / float(n)
    return lam