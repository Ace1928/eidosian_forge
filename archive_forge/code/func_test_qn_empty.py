import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
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