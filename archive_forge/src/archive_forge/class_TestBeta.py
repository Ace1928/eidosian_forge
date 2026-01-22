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
class TestBeta:

    def test_logpdf(self):
        logpdf = stats.beta.logpdf(0, 1, 0.5)
        assert_almost_equal(logpdf, -0.69314718056)
        logpdf = stats.beta.logpdf(0, 0.5, 1)
        assert_almost_equal(logpdf, np.inf)

    def test_logpdf_ticket_1866(self):
        alpha, beta = (267, 1472)
        x = np.array([0.2, 0.5, 0.6])
        b = stats.beta(alpha, beta)
        assert_allclose(b.logpdf(x).sum(), -1201.699061824062)
        assert_allclose(b.pdf(x), np.exp(b.logpdf(x)))

    def test_fit_bad_keyword_args(self):
        x = [0.1, 0.5, 0.6]
        assert_raises(TypeError, stats.beta.fit, x, floc=0, fscale=1, plate='shrimp')

    def test_fit_duplicated_fixed_parameter(self):
        x = [0.1, 0.5, 0.6]
        assert_raises(ValueError, stats.beta.fit, x, fa=0.5, fix_a=0.5)

    @pytest.mark.skipif(MACOS_INTEL, reason='Overflow, see gh-14901')
    def test_issue_12635(self):
        p, a, b = (0.9999999999997369, 75.0, 66334470.0)
        assert_allclose(stats.beta.ppf(p, a, b), 2.343620802982393e-06)

    @pytest.mark.skipif(MACOS_INTEL, reason='Overflow, see gh-14901')
    def test_issue_12794(self):
        inv_R = np.array([0.0004944464889611935, 0.0018360586912635726, 0.012266391994251835])
        count_list = np.array([10, 100, 1000])
        p = 1e-11
        inv = stats.beta.isf(p, count_list + 1, 100000 - count_list)
        assert_allclose(inv, inv_R)
        res = stats.beta.sf(inv, count_list + 1, 100000 - count_list)
        assert_allclose(res, p)

    @pytest.mark.skipif(MACOS_INTEL, reason='Overflow, see gh-14901')
    def test_issue_12796(self):
        alpha_2 = 5e-06
        count_ = np.arange(1, 20)
        nobs = 100000
        q, a, b = (1 - alpha_2, count_ + 1, nobs - count_)
        inv = stats.beta.ppf(q, a, b)
        res = stats.beta.cdf(inv, a, b)
        assert_allclose(res, 1 - alpha_2)

    def test_endpoints(self):
        a, b = (1, 0.5)
        assert_equal(stats.beta.pdf(1, a, b), np.inf)
        a, b = (0.2, 3)
        assert_equal(stats.beta.pdf(0, a, b), np.inf)
        a, b = (1, 5)
        assert_equal(stats.beta.pdf(0, a, b), 5)
        assert_equal(stats.beta.pdf(1e-310, a, b), 5)
        a, b = (5, 1)
        assert_equal(stats.beta.pdf(1, a, b), 5)
        assert_equal(stats.beta.pdf(1 - 1e-310, a, b), 5)

    @pytest.mark.xfail(IS_PYPY, reason='Does not convert boost warning')
    def test_boost_eval_issue_14606(self):
        q, a, b = (0.995, 100000000000.0, 10000000000000.0)
        with pytest.warns(RuntimeWarning):
            stats.beta.ppf(q, a, b)

    @pytest.mark.parametrize('method', [stats.beta.ppf, stats.beta.isf])
    @pytest.mark.parametrize('a, b', [(1e-310, 12.5), (12.5, 1e-310)])
    def test_beta_ppf_with_subnormal_a_b(self, method, a, b):
        p = 0.9
        try:
            method(p, a, b)
        except OverflowError:
            pass

    @pytest.mark.parametrize('a, b, ref', [(0.5, 0.5, -0.24156447527049044), (0.001, 1, -992.0922447210179), (1, 10000, -8.210440371976183), (100000, 100000, -5.377247470132859)])
    def test_entropy(self, a, b, ref):
        assert_allclose(stats.beta(a, b).entropy(), ref)

    @pytest.mark.parametrize('a, b, ref, tol', [(1, 10, -1.4025850929940458, 1e-14), (10, 20, -1.0567887388936708, 1e-13), (4000000.0, 4000000.0 + 20, -7.221686009678741, 1e-09), (5000000.0, 5000000.0 + 10, -7.333257022834638, 1e-08), (10000000000.0, 10000000000.0 + 20, -11.133707703130474, 1e-11), (1e+50, 1e+50 + 20, -57.185409562486385, 1e-15), (2, 10000000000.0, -21.448635265288925, 1e-11), (2, 1e+20, -44.47448619497938, 1e-14), (2, 1e+50, -113.55203898480075, 1e-14), (5, 10000000000.0, -20.87226777401971, 1e-10), (5, 1e+20, -43.89811870326017, 1e-14), (5, 1e+50, -112.97567149308153, 1e-14), (10, 10000000000.0, -20.489796752909477, 1e-09), (10, 1e+20, -43.51564768139993, 1e-14), (10, 1e+50, -112.59320047122131, 1e-14), (1e+20, 2, -44.47448619497938, 1e-14), (1e+20, 5, -43.89811870326017, 1e-14), (1e+50, 10, -112.59320047122131, 1e-14)])
    def test_extreme_entropy(self, a, b, ref, tol):
        assert_allclose(stats.beta(a, b).entropy(), ref, rtol=tol)