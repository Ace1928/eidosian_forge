import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root
def x0_2(n):
    x0 = np.empty([n])
    x0.fill(1 / n ** 2)
    return x0