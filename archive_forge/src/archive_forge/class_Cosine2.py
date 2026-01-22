from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
class Cosine2(CustomKernel):
    """
    Cosine2 Kernel

    K(u) = 1 + cos(2 * pi * u) between -0.5 and 0.5

    Note: this  is the same Cosine kernel that Stata uses
    """

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 1 + np.cos(2.0 * np.pi * x), h=h, domain=[-0.5, 0.5], norm=1.0)
        self._L2Norm = 1.5
        self._kernel_var = 0.03267274151216444
        self._order = 2