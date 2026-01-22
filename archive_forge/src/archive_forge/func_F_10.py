import itertools
import numpy as np
from numpy import exp
from numpy.testing import assert_, assert_equal
from scipy.optimize import root
def F_10(x, n):
    return np.log(1 + x) - x / n