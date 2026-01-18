import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_statistics_for_2d_input(self):
    stats = ddm.distance_statistics(np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float))
    assert_almost_equal(stats.test_statistic, self.test_stat_emp_exp, 0)
    assert_almost_equal(stats.distance_correlation, self.dcor_exp, 4)
    assert_almost_equal(stats.distance_covariance, self.dcov_exp, 4)
    assert_almost_equal(stats.dvar_x, self.dvar_x_exp, 4)
    assert_almost_equal(stats.dvar_y, self.dvar_y_exp, 4)
    assert_almost_equal(stats.S, self.S_exp, 4)