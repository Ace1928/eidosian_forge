import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
class TestIqrAxes:

    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10, 30))

    def test_axis0(self):
        m = scale.iqr(self.X, axis=0)
        assert_equal(m.shape, (10, 30))

    def test_axis1(self):
        m = scale.iqr(self.X, axis=1)
        assert_equal(m.shape, (40, 30))

    def test_axis2(self):
        m = scale.iqr(self.X, axis=2)
        assert_equal(m.shape, (40, 10))

    def test_axisneg1(self):
        m = scale.iqr(self.X, axis=-1)
        assert_equal(m.shape, (40, 10))