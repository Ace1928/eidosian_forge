import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
class TestLogistic:

    def test_cdf_ppf(self):
        x = np.linspace(-20, 20)
        y = stats.logistic.cdf(x)
        xx = stats.logistic.ppf(y)
        assert_allclose(x, xx)

    def test_sf_isf(self):
        x = np.linspace(-20, 20)
        y = stats.logistic.sf(x)
        xx = stats.logistic.isf(y)
        assert_allclose(x, xx)

    def test_extreme_values(self):
        p = 9.992007221626409e-16
        desired = 34.53957599234088
        assert_allclose(stats.logistic.ppf(1 - p), desired)
        assert_allclose(stats.logistic.isf(p), desired)

    def test_logpdf_basic(self):
        logp = stats.logistic.logpdf([-15, 0, 10])
        expected = [-15.000000611804547, -1.3862943611198906, -10.000090797798434]
        assert_allclose(logp, expected, rtol=1e-13)

    def test_logpdf_extreme_values(self):
        logp = stats.logistic.logpdf([800, -800])
        assert_equal(logp, [-800, -800])

    @pytest.mark.parametrize('loc_rvs,scale_rvs', [(0.4484955, 0.10216821), (0.62918191, 0.74367064)])
    def test_fit(self, loc_rvs, scale_rvs):
        data = stats.logistic.rvs(size=100, loc=loc_rvs, scale=scale_rvs)

        def func(input, data):
            a, b = input
            n = len(data)
            x1 = np.sum(np.exp((data - a) / b) / (1 + np.exp((data - a) / b))) - n / 2
            x2 = np.sum((data - a) / b * ((np.exp((data - a) / b) - 1) / (np.exp((data - a) / b) + 1))) - n
            return (x1, x2)
        expected_solution = root(func, stats.logistic._fitstart(data), args=(data,)).x
        fit_method = stats.logistic.fit(data)
        assert_allclose(fit_method, expected_solution, atol=1e-30)

    def test_fit_comp_optimizer(self):
        data = stats.logistic.rvs(size=100, loc=0.5, scale=2)
        _assert_less_or_close_loglike(stats.logistic, data)
        _assert_less_or_close_loglike(stats.logistic, data, floc=1)
        _assert_less_or_close_loglike(stats.logistic, data, fscale=1)

    @pytest.mark.parametrize('testlogcdf', [True, False])
    def test_logcdfsf_tails(self, testlogcdf):
        x = np.array([-10000, -800, 17, 50, 500])
        if testlogcdf:
            y = stats.logistic.logcdf(x)
        else:
            y = stats.logistic.logsf(-x)
        expected = [-10000.0, -800.0, -4.139937633089748e-08, -1.9287498479639178e-22, -7.124576406741286e-218]
        assert_allclose(y, expected, rtol=2e-15)

    def test_fit_gh_18176(self):
        data = np.array([-459, 37, 43, 45, 45, 48, 54, 55, 58] + [59] * 3 + [61] * 9)
        _assert_less_or_close_loglike(stats.logistic, data)