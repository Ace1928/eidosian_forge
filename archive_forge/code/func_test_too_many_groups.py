import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
def test_too_many_groups(self):
    long_groups = self.groups.reshape(-1, 1)
    groups3 = np.hstack((long_groups, long_groups, long_groups))
    assert_raises(ValueError, self.res1.get_robustcov_results, 'cluster', groups=groups3, use_correction=True, use_t=True)