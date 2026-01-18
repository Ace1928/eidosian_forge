import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import ode

    Analytical solution to the linear differential equations dy/dt = a*y.

    The solution is only valid if `a` is diagonalizable.

    Returns a 2-D array with shape (len(t), len(y0)).
    