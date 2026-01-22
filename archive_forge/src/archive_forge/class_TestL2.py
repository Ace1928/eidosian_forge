import numpy as np
from numpy.testing import assert_allclose, assert_equal
import statsmodels.base._penalties as smpen
from statsmodels.tools.numdiff import approx_fprime, approx_hess
class TestL2(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.L2()

    def test_backward_compatibility(self):
        wts = [0.5]
        pen = smpen.L2(weights=wts)
        assert_equal(pen.weights, wts)

    def test_deprecated_priority(self):
        weights = [1.0]
        pen = smpen.L2(weights=weights)
        assert_equal(pen.weights, weights)

    def test_weights_assignment(self):
        weights = [1.0, 2.0]
        pen = smpen.L2(weights=weights)
        assert_equal(pen.weights, weights)