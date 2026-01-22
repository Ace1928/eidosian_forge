import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root
def F_4(x, n):
    assert_equal(n % 3, 0)
    g = np.zeros([n])
    g[::3] = 0.6 * x[::3] + 1.6 * x[1::3] ** 3 - 7.2 * x[1::3] ** 2 + 9.6 * x[1::3] - 4.8
    g[1::3] = 0.48 * x[::3] - 0.72 * x[1::3] ** 3 + 3.24 * x[1::3] ** 2 - 4.32 * x[1::3] - x[2::3] + 0.2 * x[2::3] ** 3 + 2.16
    g[2::3] = 1.25 * x[2::3] - 0.25 * x[2::3] ** 3
    return g