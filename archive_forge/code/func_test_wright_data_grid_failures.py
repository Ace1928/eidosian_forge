import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
from scipy.special import rgamma, wright_bessel
@pytest.mark.xfail
@pytest.mark.parametrize('a, b, x, phi', grid_a_b_x_value_acc[:, :4].tolist())
def test_wright_data_grid_failures(a, b, x, phi):
    """Test cases of test_data that do not reach relative accuracy of 1e-11"""
    assert_allclose(wright_bessel(a, b, x), phi, rtol=1e-11)