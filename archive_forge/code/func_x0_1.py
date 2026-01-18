import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root
def x0_1(n):
    x0 = np.empty([n])
    x0.fill(n / (n - 1))
    return x0