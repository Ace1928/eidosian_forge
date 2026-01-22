from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
class Tricube(CustomKernel):
    """
    Tricube Kernel

    K(u) = 0.864197530864 * (1 - abs(x)**3)**3 between -1.0 and 1.0
    """

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.864197530864 * (1 - abs(x) ** 3) ** 3, h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 175.0 / 247.0
        self._kernel_var = 35.0 / 243.0
        self._order = 2