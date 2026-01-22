import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
class BrownianBridge:

    def __init__(self):
        pass

    def simulate(self, x0, x1, nobs, nrepl=1, ddt=1.0, sigma=1.0):
        nobs = nobs + 1
        dt = ddt * 1.0 / nobs
        t = np.linspace(dt, ddt - dt, nobs)
        t = np.linspace(dt, ddt, nobs)
        wm = [t / ddt, 1 - t / ddt]
        wmi = 1 - dt / (ddt - t)
        wm1 = x1 * (dt / (ddt - t))
        su = sigma * np.sqrt(t * (1 - t) / ddt)
        s = sigma * np.sqrt(dt * (ddt - t - dt) / (ddt - t))
        x = np.zeros((nrepl, nobs))
        x[:, 0] = x0
        rvs = s * np.random.normal(size=(nrepl, nobs))
        for i in range(1, nobs):
            x[:, i] = x[:, i - 1] * wmi[i] + wm1[i] + rvs[:, i]
        return (x, t, su)