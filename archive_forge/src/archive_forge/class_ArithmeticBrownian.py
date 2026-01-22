import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
class ArithmeticBrownian(AffineDiffusion):
    """
    :math::
    dx_t &= \\mu dt + \\sigma dW_t
    """

    def __init__(self, xzero, mu, sigma):
        self.xzero = xzero
        self.mu = mu
        self.sigma = sigma

    def _drift(self, *args, **kwds):
        return self.mu

    def _sig(self, *args, **kwds):
        return self.sigma

    def exactprocess(self, nobs, xzero=None, ddt=1.0, nrepl=2):
        """ddt : discrete delta t

        not tested yet
        """
        if xzero is None:
            xzero = self.xzero
        t = np.linspace(ddt, nobs * ddt, nobs)
        normrvs = np.random.normal(size=(nrepl, nobs))
        inc = self._drift + self._sigma * np.sqrt(ddt) * normrvs
        return xzero + np.cumsum(inc, 1)

    def exactdist(self, xzero, t):
        expnt = np.exp(-self.lambd * t)
        meant = self._drift * t
        stdt = self._sigma * np.sqrt(t)
        return stats.norm(loc=meant, scale=stdt)