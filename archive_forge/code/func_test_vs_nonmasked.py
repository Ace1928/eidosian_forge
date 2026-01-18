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
def test_vs_nonmasked(self):
    np.random.seed(1234567)
    outcome = np.random.randn(20, 4) + [0, 0, 1, 2]
    res1 = stats.ttest_1samp(outcome[:, 0], 1)
    res2 = mstats.ttest_1samp(outcome[:, 0], 1)
    assert_allclose(res1, res2)