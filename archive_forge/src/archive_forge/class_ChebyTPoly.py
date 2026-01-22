from scipy import stats, integrate, special
import numpy as np
class ChebyTPoly:
    """Orthonormal (for weight=1) Chebychev Polynomial on (-1,1)


    Notes
    -----
    integration requires to stay away from boundary, offsetfactor > 0
    maybe this implies that we cannot use it for densities that are > 0 at
    boundary ???

    or maybe there is a mistake close to the boundary, sometimes integration works.

    """

    def __init__(self, order):
        self.order = order
        from scipy.special import chebyt
        self.poly = chebyt(order)
        self.domain = (-1, 1)
        self.intdomain = (-1 + 1e-06, 1 - 1e-06)
        self.offsetfactor = 0.01

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x) / (1 - x ** 2) ** (1 / 4.0) / np.sqrt(np.pi)
        else:
            return self.poly(x) / (1 - x ** 2) ** (1 / 4.0) / np.sqrt(np.pi) * np.sqrt(2)