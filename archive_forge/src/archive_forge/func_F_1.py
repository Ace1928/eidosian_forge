import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root
def F_1(x, n):
    g = np.zeros([n])
    i = np.arange(2, n + 1)
    g[0] = exp(x[0] - 1) - 1
    g[1:] = i * (exp(x[1:] - 1) - x[1:])
    return g