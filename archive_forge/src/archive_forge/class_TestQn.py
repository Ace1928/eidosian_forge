import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
class TestQn:

    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.normal = standard_normal(size=40)
        cls.range = np.arange(0, 40)
        cls.exponential = np.random.exponential(size=40)
        cls.stackloss = sm.datasets.stackloss.load_pandas().data
        cls.sunspot = sm.datasets.sunspots.load_pandas().data.SUNACTIVITY

    def test_qn_naive(self):
        assert_almost_equal(scale.qn_scale(self.normal), scale._qn_naive(self.normal), DECIMAL)
        assert_almost_equal(scale.qn_scale(self.range), scale._qn_naive(self.range), DECIMAL)
        assert_almost_equal(scale.qn_scale(self.exponential), scale._qn_naive(self.exponential), DECIMAL)

    def test_qn_robustbase(self):
        assert_almost_equal(scale.qn_scale(self.range), 13.3148, DECIMAL)
        assert_almost_equal(scale.qn_scale(self.stackloss), np.array([8.87656, 8.87656, 2.21914, 4.43828]), DECIMAL)
        assert_almost_equal(scale.qn_scale(self.sunspot[0:289]), 33.50901, DECIMAL)

    def test_qn_empty(self):
        empty = np.empty(0)
        assert np.isnan(scale.qn_scale(empty))
        empty = np.empty((10, 100, 0))
        assert_equal(scale.qn_scale(empty, axis=1), np.empty((10, 0)))
        empty = np.empty((100, 100, 0, 0))
        assert_equal(scale.qn_scale(empty, axis=-1), np.empty((100, 100, 0)))
        empty = np.empty(shape=())
        with pytest.raises(ValueError):
            scale.iqr(empty)