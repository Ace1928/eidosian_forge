from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def _scalar_func_1(self, s):
    self.fcount += 1
    p = -s - s ** 3 + s ** 4
    dp = -1 - 3 * s ** 2 + 4 * s ** 3
    return (p, dp)