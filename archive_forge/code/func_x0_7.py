import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root
def x0_7(n):
    assert_equal(n % 3, 0)
    return np.array([0.001, 18, 1] * (n // 3))