from scipy import stats, integrate, special
import numpy as np
class FPoly:
    """Orthonormal (for weight=1) Fourier Polynomial on [0,1]

    orthonormal polynomial but density needs corfactor that I do not see what
    it is analytically

    parameterization on [0,1] from

    Sam Efromovich: Orthogonal series density estimation,
    2010 John Wiley & Sons, Inc. WIREs Comp Stat 2010 2 467-476


    """

    def __init__(self, order):
        self.order = order
        self.domain = (0, 1)
        self.intdomain = self.domain

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x)
        else:
            return sqr2 * np.cos(np.pi * self.order * x)