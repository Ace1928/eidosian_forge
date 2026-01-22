from scipy import stats, integrate, special
import numpy as np
class F2Poly:
    """Orthogonal (for weight=1) Fourier Polynomial on [0,pi]

    is orthogonal but first component does not square-integrate to 1
    final result seems to need a correction factor of sqrt(pi)
    _corfactor = sqrt(pi) from integrating the density

    Parameterization on [0, pi] from

    Peter Hall, Cross-Validation and the Smoothing of Orthogonal Series Density
    Estimators, JOURNAL OF MULTIVARIATE ANALYSIS 21, 189-206 (1987)

    """

    def __init__(self, order):
        self.order = order
        self.domain = (0, np.pi)
        self.intdomain = self.domain
        self.offsetfactor = 0

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x) / np.sqrt(np.pi)
        else:
            return sqr2 * np.cos(self.order * x) / np.sqrt(np.pi)