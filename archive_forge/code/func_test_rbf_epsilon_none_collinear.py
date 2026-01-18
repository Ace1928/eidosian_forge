import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def test_rbf_epsilon_none_collinear():
    x = [1, 2, 3]
    y = [4, 4, 4]
    z = [5, 6, 7]
    rbf = Rbf(x, y, z, epsilon=None)
    assert_(rbf.epsilon > 0)