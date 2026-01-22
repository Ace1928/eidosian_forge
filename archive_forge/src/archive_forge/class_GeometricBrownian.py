import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
class GeometricBrownian(AffineDiffusion):
    """Geometric Brownian Motion

    :math::
    dx_t &= \\mu x_t dt + \\sigma x_t dW_t

    $x_t $ stochastic process of Geometric Brownian motion,
    $\\mu $ is the drift,
    $\\sigma $ is the Volatility,
    $W$ is the Wiener process (Brownian motion).

    """

    def __init__(self, xzero, mu, sigma):
        self.xzero = xzero
        self.mu = mu
        self.sigma = sigma

    def _drift(self, *args, **kwds):
        x = kwds['x']
        return self.mu * x

    def _sig(self, *args, **kwds):
        x = kwds['x']
        return self.sigma * x