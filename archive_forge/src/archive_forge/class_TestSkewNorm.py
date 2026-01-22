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
class TestSkewNorm:

    def setup_method(self):
        self.rng = check_random_state(1234)

    def test_normal(self):
        x = np.linspace(-5, 5, 100)
        assert_array_almost_equal(stats.skewnorm.pdf(x, a=0), stats.norm.pdf(x))

    def test_rvs(self):
        shape = (3, 4, 5)
        x = stats.skewnorm.rvs(a=0.75, size=shape, random_state=self.rng)
        assert_equal(shape, x.shape)
        x = stats.skewnorm.rvs(a=-3, size=shape, random_state=self.rng)
        assert_equal(shape, x.shape)

    def test_moments(self):
        X = stats.skewnorm.rvs(a=4, size=int(1000000.0), loc=5, scale=2, random_state=self.rng)
        expected = [np.mean(X), np.var(X), stats.skew(X), stats.kurtosis(X)]
        computed = stats.skewnorm.stats(a=4, loc=5, scale=2, moments='mvsk')
        assert_array_almost_equal(computed, expected, decimal=2)
        X = stats.skewnorm.rvs(a=-4, size=int(1000000.0), loc=5, scale=2, random_state=self.rng)
        expected = [np.mean(X), np.var(X), stats.skew(X), stats.kurtosis(X)]
        computed = stats.skewnorm.stats(a=-4, loc=5, scale=2, moments='mvsk')
        assert_array_almost_equal(computed, expected, decimal=2)

    def test_pdf_large_x(self):
        logpdfvals = [[40, -1, -1604.8342333663986], [40, -1 / 2, -1004.1429467237419], [40, 0, -800.9189385332047], [40, 1 / 2, -800.2257913526447], [-40, -1 / 2, -800.2257913526447], [-2, 10000000.0, -200000000000019.97], [2, -10000000.0, -200000000000019.97]]
        for x, a, logpdfval in logpdfvals:
            logp = stats.skewnorm.logpdf(x, a)
            assert_allclose(logp, logpdfval, rtol=1e-08)

    def test_cdf_large_x(self):
        p = stats.skewnorm.cdf([10, 20, 30], -1)
        assert_allclose(p, np.ones(3), rtol=1e-14)
        p = stats.skewnorm.cdf(25, 2.5)
        assert_allclose(p, 1.0, rtol=1e-14)

    def test_cdf_sf_small_values(self):
        cdfvals = [[-8, 1, 3.8700350466643927e-31], [-4, 2, 8.12983991888114e-21], [-2, 5, 1.5532682678710626e-26], [-9, -1, 2.257176811907681e-19], [-10, -4, 1.523970604832105e-23]]
        for x, a, cdfval in cdfvals:
            p = stats.skewnorm.cdf(x, a)
            assert_allclose(p, cdfval, rtol=1e-08)
            p = stats.skewnorm.sf(-x, -a)
            assert_allclose(p, cdfval, rtol=1e-08)

    @pytest.mark.parametrize('a, moments', _skewnorm_noncentral_moments)
    def test_noncentral_moments(self, a, moments):
        for order, expected in enumerate(moments, start=1):
            mom = stats.skewnorm.moment(order, a)
            assert_allclose(mom, expected, rtol=1e-14)

    def test_fit(self):
        rng = np.random.default_rng(4609813989115202851)
        a, loc, scale = (-2, 3.5, 0.5)
        dist = stats.skewnorm(a, loc, scale)
        rvs = dist.rvs(size=100, random_state=rng)
        a2, loc2, scale2 = stats.skewnorm.fit(rvs, -1.5, floc=3)
        a3, loc3, scale3 = stats.skewnorm.fit(rvs, -1.6, floc=3)
        assert loc2 == loc3 == 3
        assert a2 != a3
        a4, loc4, scale4 = stats.skewnorm.fit(rvs, 3, fscale=3, method='mm')
        assert scale4 == 3
        dist4 = stats.skewnorm(a4, loc4, scale4)
        res = dist4.stats(moments='ms')
        ref = (np.mean(rvs), stats.skew(rvs))
        assert_allclose(res, ref)
        rvs2 = stats.pareto.rvs(1, size=100, random_state=rng)
        res = stats.skewnorm.fit(rvs2)
        assert np.all(np.isfinite(res))
        a5, loc5, scale5 = stats.skewnorm.fit(rvs2, method='mm')
        assert np.isinf(a5)
        m, v = (np.mean(rvs2), np.var(rvs2))
        assert_allclose(m, loc5 + scale5 * np.sqrt(2 / np.pi))
        assert_allclose(v, scale5 ** 2 * (1 - 2 / np.pi))
        a6p, loc6p, scale6p = stats.skewnorm.fit(rvs, method='mle')
        a6m, loc6m, scale6m = stats.skewnorm.fit(-rvs, method='mle')
        assert_allclose([a6m, loc6m, scale6m], [-a6p, -loc6p, scale6p])
        a7p, loc7p, scale7p = stats.skewnorm.fit(rvs, method='mm')
        a7m, loc7m, scale7m = stats.skewnorm.fit(-rvs, method='mm')
        assert_allclose([a7m, loc7m, scale7m], [-a7p, -loc7p, scale7p])

    def test_fit_gh19332(self):
        x = np.array([-5, -1, 1 / 100000] + 12 * [1] + [5])
        params = stats.skewnorm.fit(x)
        res = stats.skewnorm.nnlf(params, x)
        params_super = stats.skewnorm.fit(x, superfit=True)
        ref = stats.skewnorm.nnlf(params_super, x)
        assert res < ref - 0.5
        rng = np.random.default_rng(9842356982345693637)
        bounds = {'a': (-5, 5), 'loc': (-10, 10), 'scale': (1e-16, 10)}

        def optimizer(fun, bounds):
            return differential_evolution(fun, bounds, seed=rng)
        fit_result = stats.fit(stats.skewnorm, x, bounds, optimizer=optimizer)
        np.testing.assert_allclose(params, fit_result.params, rtol=0.0001)