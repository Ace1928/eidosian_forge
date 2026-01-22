import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
class Diffusion:
    """Wiener Process, Brownian Motion with mu=0 and sigma=1
    """

    def __init__(self):
        pass

    def simulateW(self, nobs=100, T=1, dt=None, nrepl=1):
        """generate sample of Wiener Process
        """
        dt = T * 1.0 / nobs
        t = np.linspace(dt, 1, nobs)
        dW = np.sqrt(dt) * np.random.normal(size=(nrepl, nobs))
        W = np.cumsum(dW, 1)
        self.dW = dW
        return (W, t)

    def expectedsim(self, func, nobs=100, T=1, dt=None, nrepl=1):
        """get expectation of a function of a Wiener Process by simulation

        initially test example from
        """
        W, t = self.simulateW(nobs=nobs, T=T, dt=dt, nrepl=nrepl)
        U = func(t, W)
        Umean = U.mean(0)
        return (U, Umean, t)