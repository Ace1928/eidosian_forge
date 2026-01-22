import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import ode
def _linear_jac(t, y, a):
    """Jacobian of a * y is a."""
    return a