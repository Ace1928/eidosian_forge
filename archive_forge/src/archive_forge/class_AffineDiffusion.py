import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
class AffineDiffusion(Diffusion):
    """

    differential equation:

    :math::
    dx_t = f(t,x)dt + \\sigma(t,x)dW_t

    integral:

    :math::
    x_T = x_0 + \\int_{0}^{T}f(t,S)dt + \\int_0^T  \\sigma(t,S)dW_t

    TODO: check definition, affine, what about jump diffusion?

    """

    def __init__(self):
        pass

    def sim(self, nobs=100, T=1, dt=None, nrepl=1):
        W, t = self.simulateW(nobs=nobs, T=T, dt=dt, nrepl=nrepl)
        dx = self._drift() + self._sig() * W
        x = np.cumsum(dx, 1)
        xmean = x.mean(0)
        return (x, xmean, t)

    def simEM(self, xzero=None, nobs=100, T=1, dt=None, nrepl=1, Tratio=4):
        """

        from Higham 2001

        TODO: reverse parameterization to start with final nobs and DT
        TODO: check if I can skip the loop using my way from exactprocess
              problem might be Winc (reshape into 3d and sum)
        TODO: (later) check memory efficiency for large simulations
        """
        nobs = nobs * Tratio
        if xzero is None:
            xzero = self.xzero
        if dt is None:
            dt = T * 1.0 / nobs
        W, t = self.simulateW(nobs=nobs, T=T, dt=dt, nrepl=nrepl)
        dW = self.dW
        t = np.linspace(dt, 1, nobs)
        Dt = Tratio * dt
        L = nobs / Tratio
        Xem = np.zeros((nrepl, L))
        Xtemp = xzero
        Xem[:, 0] = xzero
        for j in np.arange(1, L):
            Winc = np.sum(dW[:, np.arange(Tratio * (j - 1) + 1, Tratio * j)], 1)
            Xtemp = Xtemp + self._drift(x=Xtemp) + self._sig(x=Xtemp) * Winc
            Xem[:, j] = Xtemp
        return Xem