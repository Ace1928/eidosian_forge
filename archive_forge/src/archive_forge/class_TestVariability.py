import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
class TestVariability:
    """  Comparison numbers are found using R v.1.5.1
         note that length(testcase) = 4
    """
    testcase = ma.fix_invalid([1, 2, 3, 4, np.nan])

    def test_sem(self):
        y = mstats.sem(self.testcase)
        assert_almost_equal(y, 0.6454972244)
        n = self.testcase.count()
        assert_allclose(mstats.sem(self.testcase, ddof=0) * np.sqrt(n / (n - 2)), mstats.sem(self.testcase, ddof=2))

    def test_zmap(self):
        y = mstats.zmap(self.testcase, self.testcase)
        desired_unmaskedvals = [-1.3416407864999, -0.44721359549996, 0.44721359549996, 1.3416407864999]
        assert_array_almost_equal(desired_unmaskedvals, y.data[y.mask == False], decimal=12)

    def test_zscore(self):
        y = mstats.zscore(self.testcase)
        desired = ma.fix_invalid([-1.3416407864999, -0.44721359549996, 0.44721359549996, 1.3416407864999, np.nan])
        assert_almost_equal(desired, y, decimal=12)