import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
class OUprocess(AffineDiffusion):
    """Ornstein-Uhlenbeck

    :math::
      dx_t&=\\lambda(\\mu - x_t)dt+\\sigma dW_t

    mean reverting process



    TODO: move exact higher up in class hierarchy
    """

    def __init__(self, xzero, mu, lambd, sigma):
        self.xzero = xzero
        self.lambd = lambd
        self.mu = mu
        self.sigma = sigma

    def _drift(self, *args, **kwds):
        x = kwds['x']
        return self.lambd * (self.mu - x)

    def _sig(self, *args, **kwds):
        x = kwds['x']
        return self.sigma * x

    def exact(self, xzero, t, normrvs):
        expnt = np.exp(-self.lambd * t)
        return xzero * expnt + self.mu * (1 - expnt) + self.sigma * np.sqrt((1 - expnt * expnt) / 2.0 / self.lambd) * normrvs

    def exactprocess(self, xzero, nobs, ddt=1.0, nrepl=2):
        """ddt : discrete delta t

        should be the same as an AR(1)
        not tested yet
        # after writing this I saw the same use of lfilter in sitmo
        """
        t = np.linspace(ddt, nobs * ddt, nobs)
        expnt = np.exp(-self.lambd * t)
        expddt = np.exp(-self.lambd * ddt)
        normrvs = np.random.normal(size=(nrepl, nobs))
        from scipy import signal
        inc = self.mu * (1 - expddt) + self.sigma * np.sqrt((1 - expddt * expddt) / 2.0 / self.lambd) * normrvs
        return signal.lfilter([1.0], [1.0, -expddt], inc)

    def exactdist(self, xzero, t):
        expnt = np.exp(-self.lambd * t)
        meant = xzero * expnt + self.mu * (1 - expnt)
        stdt = self.sigma * np.sqrt((1 - expnt * expnt) / 2.0 / self.lambd)
        from scipy import stats
        return stats.norm(loc=meant, scale=stdt)

    def fitls(self, data, dt):
        """assumes data is 1d, univariate time series
        formula from sitmo
        """
        nobs = len(data) - 1
        exog = np.column_stack((np.ones(nobs), data[:-1]))
        parest, res, rank, sing = np.linalg.lstsq(exog, data[1:], rcond=-1)
        const, slope = parest
        errvar = res / (nobs - 2.0)
        lambd = -np.log(slope) / dt
        sigma = np.sqrt(-errvar * 2.0 * np.log(slope) / (1 - slope ** 2) / dt)
        mu = const / (1 - slope)
        return (mu, lambd, sigma)