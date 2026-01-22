import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
class TestBoxcoxNormplot:

    def setup_method(self):
        self.x = _old_loggamma_rvs(5, size=500, random_state=7654321) + 5

    def test_basic(self):
        N = 5
        lmbdas, ppcc = stats.boxcox_normplot(self.x, -10, 10, N=N)
        ppcc_expected = [0.57783375, 0.83610988, 0.97524311, 0.99756057, 0.95843297]
        assert_allclose(lmbdas, np.linspace(-10, 10, num=N))
        assert_allclose(ppcc, ppcc_expected)

    @pytest.mark.skipif(not have_matplotlib, reason='no matplotlib')
    def test_plot_kwarg(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.boxcox_normplot(self.x, -20, 20, plot=plt)
        fig.delaxes(ax)
        ax = fig.add_subplot(111)
        stats.boxcox_normplot(self.x, -20, 20, plot=ax)
        plt.close()

    def test_invalid_inputs(self):
        assert_raises(ValueError, stats.boxcox_normplot, self.x, 1, 0)
        assert_raises(ValueError, stats.boxcox_normplot, [-1, 1], 0, 1)

    def test_empty(self):
        assert_(stats.boxcox_normplot([], 0, 1).size == 0)