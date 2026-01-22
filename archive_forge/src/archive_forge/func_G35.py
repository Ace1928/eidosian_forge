import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def G35(h2, k2):
    res = 16 * h2 ** 4 + 6 * k2 ** 4 - 28 * h2 ** 3 * k2 - 12 * h2 * k2 ** 3 + 34 * h2 ** 2 * k2 ** 2 + sqrt(4 * h2 ** 2 + k2 ** 2 - h2 * k2) * (8 * h2 ** 3 + 6 * k2 ** 3 - 13 * h2 ** 2 * k2 - 9 * h2 * k2 ** 2)
    return 16 * pi / 13125 * k2 * (k2 - h2) * res