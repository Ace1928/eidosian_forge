import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
class TestHyperu:

    def test_negative_x(self):
        a, b, x = np.meshgrid([-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1], np.linspace(-100, -1, 10))
        assert np.all(np.isnan(sc.hyperu(a, b, x)))

    def test_special_cases(self):
        assert sc.hyperu(0, 1, 1) == 1.0

    @pytest.mark.parametrize('a', [0.5, 1, np.nan])
    @pytest.mark.parametrize('b', [1, 2, np.nan])
    @pytest.mark.parametrize('x', [0.25, 3, np.nan])
    def test_nan_inputs(self, a, b, x):
        assert np.isnan(sc.hyperu(a, b, x)) == np.any(np.isnan([a, b, x]))