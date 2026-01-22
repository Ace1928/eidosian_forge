import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
class TestScaledExp1:

    @pytest.mark.parametrize('x, expected', [(0, 0), (np.inf, 1)])
    def test_limits(self, x, expected):
        y = sc._ufuncs._scaled_exp1(x)
        assert y == expected

    @pytest.mark.parametrize('x, expected', [(1e-25, 5.698741165994961e-24), (0.1, 0.20146425447084518), (0.9995, 0.5962509885831002), (1.0, 0.5963473623231941), (1.0005, 0.5964436833238044), (2.5, 0.7588145912149602), (10.0, 0.9156333393978808), (100.0, 0.9901942286733019), (500.0, 0.9980079523802055), (1000.0, 0.9990019940238807), (1249.5, 0.9992009578306811), (1250.0, 0.9992012769377913), (1250.25, 0.9992014363957858), (2000.0, 0.9995004992514963), (10000.0, 0.9999000199940024), (10000000000.0, 0.9999999999), (1000000000000000.0, 0.999999999999999)])
    def test_scaled_exp1(self, x, expected):
        y = sc._ufuncs._scaled_exp1(x)
        assert_allclose(y, expected, rtol=2e-15)