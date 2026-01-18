import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
def test_param_unpacking():
    expected = np.array([2.0, 3, 0, 1])
    pp = ProbPlot(np.empty(100), dist=stats.beta(2, 3))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(2, b=3))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(a=2, b=3))
    assert_equal(pp.fit_params, expected)
    expected = np.array([2.0, 3, 4, 1])
    pp = ProbPlot(np.empty(100), stats.beta(2, 3, 4))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(a=2, b=3, loc=4))
    assert_equal(pp.fit_params, expected)
    expected = np.array([2.0, 3, 4, 5])
    pp = ProbPlot(np.empty(100), stats.beta(2, 3, 4, 5))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(2, 3, 4, scale=5))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(2, 3, loc=4, scale=5))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(2, b=3, loc=4, scale=5))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(a=2, b=3, loc=4, scale=5))
    assert_equal(pp.fit_params, expected)