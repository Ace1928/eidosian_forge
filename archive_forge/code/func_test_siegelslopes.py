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
def test_siegelslopes():
    y = 2 * np.arange(10) + 0.5
    assert_equal(mstats.siegelslopes(y), (2.0, 0.5))
    assert_equal(mstats.siegelslopes(y, method='separate'), (2.0, 0.5))
    x = 2 * np.arange(10)
    y = 5 * x - 3.0
    assert_equal(mstats.siegelslopes(y, x), (5.0, -3.0))
    assert_equal(mstats.siegelslopes(y, x, method='separate'), (5.0, -3.0))
    y[:4] = 1000
    assert_equal(mstats.siegelslopes(y, x), (5.0, -3.0))
    x = np.arange(10)
    y = -2.3 + 0.3 * x + stats.norm.rvs(size=10, random_state=231)
    slope_ols, intercept_ols, _, _, _ = stats.linregress(x, y)
    slope, intercept = mstats.siegelslopes(y, x)
    assert_allclose(slope, slope_ols, rtol=0.1)
    assert_allclose(intercept, intercept_ols, rtol=0.1)
    slope, intercept = mstats.siegelslopes(y, x, method='separate')
    assert_allclose(slope, slope_ols, rtol=0.1)
    assert_allclose(intercept, intercept_ols, rtol=0.1)