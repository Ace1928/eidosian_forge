from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
@property
def L2Norm(self):
    """Returns the integral of the square of the kernal from -inf to inf"""
    if self._L2Norm is None:
        L2Func = lambda x: (self.norm_const * self._shape(x)) ** 2
        if self.domain is None:
            self._L2Norm = scipy.integrate.quad(L2Func, -inf, inf)[0]
        else:
            self._L2Norm = scipy.integrate.quad(L2Func, self.domain[0], self.domain[1])[0]
    return self._L2Norm