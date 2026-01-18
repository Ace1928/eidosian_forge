import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.special import log_ndtr, ndtri_exp
from scipy.special._testutils import assert_func_equal
@pytest.mark.parametrize('test_input', [-10.0, -100.0, -10000000000.0, -1e+20, -np.finfo(float).max])
def test_very_small_arg(self, test_input, uniform_random_points):
    scale = test_input
    points = scale * (0.5 * uniform_random_points + 0.5)
    assert_func_equal(log_ndtr_ndtri_exp, lambda y: y, points, rtol=1e-14, nan_ok=True)