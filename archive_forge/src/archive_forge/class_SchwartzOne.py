import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
class SchwartzOne(ExactDiffusion):
    """the Schwartz type 1 stochastic process

    :math::
    dx_t = \\kappa (\\mu - \\ln x_t) x_t dt + \\sigma x_tdW \\

    The Schwartz type 1 process is a log of the Ornstein-Uhlenbeck stochastic
    process.

    """

    def __init__(self, xzero, mu, kappa, sigma):
        self.xzero = xzero
        self.mu = mu
        self.kappa = kappa
        self.lambd = kappa
        self.sigma = sigma

    def _exactconst(self, expnt):
        return (1 - expnt) * (self.mu - self.sigma ** 2 / 2.0 / self.kappa)

    def _exactstd(self, expnt):
        return self.sigma * np.sqrt((1 - expnt * expnt) / 2.0 / self.kappa)

    def exactprocess(self, xzero, nobs, ddt=1.0, nrepl=2):
        """uses exact solution for log of process
        """
        lnxzero = np.log(xzero)
        lnx = super(self.__class__, self).exactprocess(xzero, nobs, ddt=ddt, nrepl=nrepl)
        return np.exp(lnx)

    def exactdist(self, xzero, t):
        expnt = np.exp(-self.lambd * t)
        meant = np.log(xzero) * expnt + self._exactconst(expnt)
        stdt = self._exactstd(expnt)
        return stats.lognorm(loc=meant, scale=stdt)

    def fitls(self, data, dt):
        """assumes data is 1d, univariate time series
        formula from sitmo
        """
        nobs = len(data) - 1
        exog = np.column_stack((np.ones(nobs), np.log(data[:-1])))
        parest, res, rank, sing = np.linalg.lstsq(exog, np.log(data[1:]), rcond=-1)
        const, slope = parest
        errvar = res / (nobs - 2.0)
        kappa = -np.log(slope) / dt
        sigma = np.sqrt(errvar * kappa / (1 - np.exp(-2 * kappa * dt)))
        mu = const / (1 - np.exp(-kappa * dt)) + sigma ** 2 / 2.0 / kappa
        if np.shape(mu) == (1,):
            mu = mu[0]
        if np.shape(sigma) == (1,):
            sigma = sigma[0]
        return (mu, kappa, sigma)