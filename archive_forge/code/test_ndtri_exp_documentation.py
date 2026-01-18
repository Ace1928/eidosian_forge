import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.special import log_ndtr, ndtri_exp
from scipy.special._testutils import assert_func_equal
Tests that ndtri_exp is sufficiently close to an inverse of log_ndtr.

    We have separate tests for the five intervals (-inf, -10),
    [-10, -2), [-2, -0.14542), [-0.14542, -1e-6), and [-1e-6, 0).
    ndtri_exp(y) is computed in three different ways depending on if y
    is in (-inf, -2), [-2, log(1 - exp(-2))], or [log(1 - exp(-2), 0).
    Each of these intervals is given its own test with two additional tests
    for handling very small values and values very close to zero.
    