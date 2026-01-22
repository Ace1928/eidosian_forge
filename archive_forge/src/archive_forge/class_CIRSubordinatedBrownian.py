import numpy as np
import matplotlib.pyplot as plt
class CIRSubordinatedBrownian:
    """CIR subordinated Brownian Motion
    """

    def __init__(self):
        pass

    def simulate(self, m, kappa, T_dot, lambd, sigma, ts, nrepl):
        T = ts[-1]
        nobs = len(ts)
        dtarr = np.zeros(nobs)
        dtarr[0] = ts[0] - 0
        dtarr[1:] = np.diff(ts)
        DXs = np.zeros((nrepl, nobs))
        dB = np.sqrt(dtarr) * np.random.randn(nrepl, nobs)
        yt = 1.0
        dXs = np.zeros((nrepl, nobs))
        dtaus = np.zeros((nrepl, nobs))
        y = np.zeros((nrepl, nobs))
        for t in range(nobs):
            dt = dtarr[t]
            dy = kappa * (T_dot - yt) * dt + lambd * np.sqrt(yt) * dB[:, t]
            yt = np.maximum(yt + dy, 1e-10)
            dtau = np.maximum(yt * dt, 1e-06)
            dX = np.random.normal(loc=m * dtau, scale=sigma * np.sqrt(dtau))
            y[:, t] = yt
            dtaus[:, t] = dtau
            dXs[:, t] = dX
        tau = np.cumsum(dtaus, 1)
        x = np.cumsum(dXs, 1)
        return (x, tau, y)