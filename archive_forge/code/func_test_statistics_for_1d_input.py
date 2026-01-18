import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_statistics_for_1d_input(self):
    x = np.array(range(1, 21), dtype=float)
    y = x + np.log(x)
    stats = ddm.distance_statistics(x, y)
    assert_almost_equal(stats.test_statistic, 398.94623, 5)
    assert_almost_equal(stats.distance_correlation, 0.9996107, 4)
    assert_almost_equal(stats.distance_covariance, 4.4662414, 4)
    assert_almost_equal(stats.dvar_x, 4.2294799, 4)
    assert_almost_equal(stats.dvar_y, 4.7199304, 4)
    assert_almost_equal(stats.S, 49.8802, 4)