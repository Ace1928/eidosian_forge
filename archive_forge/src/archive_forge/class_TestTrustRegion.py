from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
class TestTrustRegion:

    def test_intersect(self):
        Delta = 1.0
        x = np.zeros(3)
        s = np.array([1.0, 0.0, 0.0])
        t_neg, t_pos = intersect_trust_region(x, s, Delta)
        assert_equal(t_neg, -1)
        assert_equal(t_pos, 1)
        s = np.array([-1.0, 1.0, -1.0])
        t_neg, t_pos = intersect_trust_region(x, s, Delta)
        assert_allclose(t_neg, -3 ** (-0.5))
        assert_allclose(t_pos, 3 ** (-0.5))
        x = np.array([0.5, -0.5, 0])
        s = np.array([0, 0, 1.0])
        t_neg, t_pos = intersect_trust_region(x, s, Delta)
        assert_allclose(t_neg, -2 ** (-0.5))
        assert_allclose(t_pos, 2 ** (-0.5))
        x = np.ones(3)
        assert_raises(ValueError, intersect_trust_region, x, s, Delta)
        x = np.zeros(3)
        s = np.zeros(3)
        assert_raises(ValueError, intersect_trust_region, x, s, Delta)