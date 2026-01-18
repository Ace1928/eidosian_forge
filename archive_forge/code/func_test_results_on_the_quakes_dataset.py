import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_results_on_the_quakes_dataset(self):
    """
        R code:
        ------

        > data("quakes")
        > x = quakes[1:50, 1:3]
        > y = quakes[51:100, 1:3]
        > dcov.test(x, y, R=200)

            dCov independence test (permutation test)

        data:  index 1, replicates 200
        nV^2 = 45046, p-value = 0.4577
        sample estimates:
            dCov
        30.01526
        """
    try:
        quakes = get_rdataset('quakes').data.values[:, :3]
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTPError or URLError, these are random')
    x = np.asarray(quakes[:50], dtype=float)
    y = np.asarray(quakes[50:100], dtype=float)
    stats = ddm.distance_statistics(x, y)
    assert_almost_equal(np.round(stats.test_statistic), 45046, 0)
    assert_almost_equal(stats.distance_correlation, 0.1894193, 4)
    assert_almost_equal(stats.distance_covariance, 30.01526, 4)
    assert_almost_equal(stats.dvar_x, 170.1702, 4)
    assert_almost_equal(stats.dvar_y, 147.5545, 4)
    assert_almost_equal(stats.S, 52265, 0)
    test_statistic, _, method = ddm.distance_covariance_test(x, y, B=199)
    assert_almost_equal(np.round(test_statistic), 45046, 0)
    assert method == 'emp'