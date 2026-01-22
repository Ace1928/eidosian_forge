import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
class ExactDiffusion(AffineDiffusion):
    """Diffusion that has an exact integral representation

    this is currently mainly for geometric, log processes

    """

    def __init__(self):
        pass

    def exactprocess(self, xzero, nobs, ddt=1.0, nrepl=2):
        """ddt : discrete delta t



        should be the same as an AR(1)
        not tested yet
        """
        t = np.linspace(ddt, nobs * ddt, nobs)
        expddt = np.exp(-self.lambd * ddt)
        normrvs = np.random.normal(size=(nrepl, nobs))
        inc = self._exactconst(expddt) + self._exactstd(expddt) * normrvs
        return signal.lfilter([1.0], [1.0, -expddt], inc)

    def exactdist(self, xzero, t):
        expnt = np.exp(-self.lambd * t)
        meant = xzero * expnt + self._exactconst(expnt)
        stdt = self._exactstd(expnt)
        return stats.norm(loc=meant, scale=stdt)