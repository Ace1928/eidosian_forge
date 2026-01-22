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
class TestPpccPlot:

    def setup_method(self):
        self.x = _old_loggamma_rvs(5, size=500, random_state=7654321) + 5

    def test_basic(self):
        N = 5
        svals, ppcc = stats.ppcc_plot(self.x, -10, 10, N=N)
        ppcc_expected = [0.21139644, 0.21384059, 0.98766719, 0.97980182, 0.93519298]
        assert_allclose(svals, np.linspace(-10, 10, num=N))
        assert_allclose(ppcc, ppcc_expected)

    def test_dist(self):
        svals1, ppcc1 = stats.ppcc_plot(self.x, -10, 10, dist='tukeylambda')
        svals2, ppcc2 = stats.ppcc_plot(self.x, -10, 10, dist=stats.tukeylambda)
        assert_allclose(svals1, svals2, rtol=1e-20)
        assert_allclose(ppcc1, ppcc2, rtol=1e-20)
        svals3, ppcc3 = stats.ppcc_plot(self.x, -10, 10)
        assert_allclose(svals1, svals3, rtol=1e-20)
        assert_allclose(ppcc1, ppcc3, rtol=1e-20)

    @pytest.mark.skipif(not have_matplotlib, reason='no matplotlib')
    def test_plot_kwarg(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.ppcc_plot(self.x, -20, 20, plot=plt)
        fig.delaxes(ax)
        ax = fig.add_subplot(111)
        stats.ppcc_plot(self.x, -20, 20, plot=ax)
        plt.close()

    def test_invalid_inputs(self):
        assert_raises(ValueError, stats.ppcc_plot, self.x, 1, 0)
        assert_raises(ValueError, stats.ppcc_plot, [1, 2, 3], 0, 1, dist='plate_of_shrimp')

    def test_empty(self):
        svals, ppcc = stats.ppcc_plot([], 0, 1)
        assert_allclose(svals, np.linspace(0, 1, num=80))
        assert_allclose(ppcc, np.zeros(80, dtype=float))