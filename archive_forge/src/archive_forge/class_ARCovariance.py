import numpy as np
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.moment_helpers import cov2corr
class ARCovariance:
    """
    experimental class for Covariance of AR process
    classmethod? staticmethods?
    """

    def __init__(self, ar=None, ar_coefs=None, sigma=1.0):
        if ar is not None:
            self.ar = ar
            self.ar_coefs = -ar[1:]
            self.k_lags = len(ar)
        elif ar_coefs is not None:
            self.arcoefs = ar_coefs
            self.ar = np.hstack(([1], -ar_coefs))
            self.k_lags = len(self.ar)

    @classmethod
    def fit(cls, cov, order, **kwds):
        rho, sigma = yule_walker_acov(cov, order=order, **kwds)
        return cls(ar_coefs=rho)

    def whiten(self, x):
        return whiten_ar(x, self.ar_coefs, order=self.order)

    def corr(self, k_vars=None):
        if k_vars is None:
            k_vars = len(self.ar)
        return corr_ar(k_vars, self.ar)

    def cov(self, k_vars=None):
        return cov2corr(self.corr(k_vars=None), self.sigma)