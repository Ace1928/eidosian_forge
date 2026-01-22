import numpy as np
import matplotlib.pyplot as plt
class VG:
    """variance gamma process
    """

    def __init__(self):
        pass

    def simulate(self, m, s, kappa, ts, nrepl):
        T = len(ts)
        dXs = np.zeros((nrepl, T))
        for t in range(T):
            dt = ts[1] - 0
            if t > 1:
                dt = ts[t] - ts[t - 1]
            d_tau = kappa * np.random.gamma(dt / kappa, 1.0, size=nrepl)
            dX = np.random.normal(loc=m * d_tau, scale=1e-06 + s * np.sqrt(d_tau))
            dXs[:, t] = dX
        x = np.cumsum(dXs, 1)
        return x