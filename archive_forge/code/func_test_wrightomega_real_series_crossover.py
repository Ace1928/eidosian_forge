import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
import scipy.special as sc
from scipy.special._testutils import assert_func_equal
def test_wrightomega_real_series_crossover():
    desired_error = 2 * np.finfo(float).eps
    crossover = 1e+20
    x_before_crossover = np.nextafter(crossover, -np.inf)
    x_after_crossover = np.nextafter(crossover, np.inf)
    desired_before_crossover = 9.999999999999998e+19
    desired_after_crossover = 1.0000000000000002e+20
    assert_allclose(sc.wrightomega(x_before_crossover), desired_before_crossover, atol=0, rtol=desired_error)
    assert_allclose(sc.wrightomega(x_after_crossover), desired_after_crossover, atol=0, rtol=desired_error)