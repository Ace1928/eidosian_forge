import numpy as np
import matplotlib.pyplot as plt
class Heston:
    """Heston Stochastic Volatility
    """

    def __init__(self):
        pass

    def simulate(self, m, kappa, eta, lambd, r, ts, nrepl, tratio=1.0):
        T = ts[-1]
        nobs = len(ts)
        dt = np.zeros(nobs)
        dt[0] = ts[0] - 0
        dt[1:] = np.diff(ts)
        DXs = np.zeros((nrepl, nobs))
        dB_1 = np.sqrt(dt) * np.random.randn(nrepl, nobs)
        dB_2u = np.sqrt(dt) * np.random.randn(nrepl, nobs)
        dB_2 = r * dB_1 + np.sqrt(1 - r ** 2) * dB_2u
        vt = eta * np.ones(nrepl)
        v = []
        dXs = np.zeros((nrepl, nobs))
        vts = np.zeros((nrepl, nobs))
        for t in range(nobs):
            dv = kappa * (eta - vt) * dt[t] + lambd * np.sqrt(vt) * dB_2[:, t]
            dX = m * dt[t] + np.sqrt(vt * dt[t]) * dB_1[:, t]
            vt = vt + dv
            vts[:, t] = vt
            dXs[:, t] = dX
        x = np.cumsum(dXs, 1)
        return (x, vts)