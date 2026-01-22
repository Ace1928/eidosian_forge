from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time
class g(sympy.Function):
    """Helper function g according to Wright (1935)

        g(n, rho, v) = (1 + (rho+2)/3 * v + (rho+2)*(rho+3)/(2*3) * v^2 + ...)

        Note: Wright (1935) uses square root of above definition.
        """
    nargs = 3

    @classmethod
    def eval(cls, n, rho, v):
        if not n >= 0:
            raise ValueError('must have n >= 0')
        elif n == 0:
            return 1
        else:
            return g(n - 1, rho, v) + gammasimp(gamma(rho + 2 + n) / gamma(rho + 2)) / gammasimp(gamma(3 + n) / gamma(3)) * v ** n