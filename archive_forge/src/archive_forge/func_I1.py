import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def I1(h2, k2, s):
    res = ellip_harm_2(h2, k2, 1, 1, s) / (3 * ellip_harm(h2, k2, 1, 1, s)) + ellip_harm_2(h2, k2, 1, 2, s) / (3 * ellip_harm(h2, k2, 1, 2, s)) + ellip_harm_2(h2, k2, 1, 3, s) / (3 * ellip_harm(h2, k2, 1, 3, s))
    return res