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
class TestBetaPrime:
    cdf_vals = [(1e+22, 100.0, 0.05, 0.8973027435427167), (10000000000.0, 100.0, 0.05, 0.5911548582766262), (100000000.0, 0.05, 0.1, 0.9467768090820048), (100000000.0, 100.0, 0.05, 0.4852944858726726), (1e-10, 0.05, 0.1, 0.21238845427095), (1e-10, 1.5, 1.5, 1.697652726007973e-15), (1e-10, 0.05, 100.0, 0.40884514172337383), (1e-22, 0.05, 0.1, 0.053349567649287326), (1e-22, 1.5, 1.5, 1.6976527263135503e-33), (1e-22, 0.05, 100.0, 0.10269725645728331), (1e-100, 0.05, 0.1, 6.7163126421919795e-06), (1e-100, 1.5, 1.5, 1.6976527263135503e-150), (1e-100, 0.05, 100.0, 1.2928818587561651e-05)]

    def test_logpdf(self):
        alpha, beta = (267, 1472)
        x = np.array([0.2, 0.5, 0.6])
        b = stats.betaprime(alpha, beta)
        assert_(np.isfinite(b.logpdf(x)).all())
        assert_allclose(b.pdf(x), np.exp(b.logpdf(x)))

    def test_cdf(self):
        x = stats.betaprime.cdf(0, 0.2, 0.3)
        assert_equal(x, 0.0)
        alpha, beta = (267, 1472)
        x = np.array([0.2, 0.5, 0.6])
        cdfs = stats.betaprime.cdf(x, alpha, beta)
        assert_(np.isfinite(cdfs).all())
        gen_cdf = stats.rv_continuous._cdf_single
        cdfs_g = [gen_cdf(stats.betaprime, val, alpha, beta) for val in x]
        assert_allclose(cdfs, cdfs_g, atol=0, rtol=2e-12)

    @pytest.mark.parametrize('p, a, b, expected', [(0.01, 1.25, 2.5, 0.01080162700956614), (1e-12, 1.25, 2.5, 1.0610141996279122e-10), (1e-18, 1.25, 2.5, 1.6815941817974941e-15), (1e-17, 0.25, 7.0, 1.0179194531881782e-69), (0.375, 0.25, 7.0, 0.002036820346115211), (0.9978811466052919, 0.05, 0.1, 1.0000000000001218e+22)])
    def test_ppf(self, p, a, b, expected):
        x = stats.betaprime.ppf(p, a, b)
        assert_allclose(x, expected, rtol=1e-14)

    @pytest.mark.parametrize('x, a, b, p', cdf_vals)
    def test_ppf_gh_17631(self, x, a, b, p):
        assert_allclose(stats.betaprime.ppf(p, a, b), x, rtol=1e-14)

    @pytest.mark.parametrize('x, a, b, expected', cdf_vals + [(10000000000.0, 1.5, 1.5, 0.9999999999999983), (10000000000.0, 0.05, 0.1, 0.9664184367890859), (1e+22, 0.05, 0.1, 0.9978811466052919)])
    def test_cdf_gh_17631(self, x, a, b, expected):
        assert_allclose(stats.betaprime.cdf(x, a, b), expected, rtol=1e-14)

    @pytest.mark.parametrize('x, a, b, expected', [(1e+50, 0.05, 0.1, 0.9999966641709545), (1e+50, 100.0, 0.05, 0.995925162631006)])
    def test_cdf_extreme_tails(self, x, a, b, expected):
        y = stats.betaprime.cdf(x, a, b)
        assert y < 1.0
        assert_allclose(y, expected, rtol=2e-05)

    def test_sf(self):
        a = [5, 4, 2, 0.05, 0.05, 0.05, 0.05, 100.0, 100.0, 0.05, 0.05, 0.05, 1.5, 1.5]
        b = [3, 2, 1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 100.0, 100.0, 100.0, 1.5, 1.5]
        x = [10000000000.0, 1e+20, 1e+30, 1e+22, 1e-10, 1e-22, 1e-100, 1e+22, 10000000000.0, 1e-10, 1e-22, 1e-100, 10000000000.0, 1e-10]
        ref = [3.4999999979e-29, 9.999999999994357e-40, 1.9999999999999998e-30, 0.0021188533947081017, 0.78761154572905, 0.9466504323507127, 0.9999932836873578, 0.10269725645728331, 0.40884514172337383, 0.5911548582766262, 0.8973027435427167, 0.9999870711814124, 1.6976527260079727e-15, 0.9999999999999983]
        sf_values = stats.betaprime.sf(x, a, b)
        assert_allclose(sf_values, ref, rtol=1e-12)

    def test_fit_stats_gh18274(self):
        stats.betaprime.fit([0.1, 0.25, 0.3, 1.2, 1.6], floc=0, fscale=1)
        stats.betaprime(a=1, b=1).stats('mvsk')

    def test_moment_gh18634(self):
        ref = [np.inf, 0.867096912929055]
        res = stats.betaprime(2, [4.2, 7.1]).moment(5)
        assert_allclose(res, ref)