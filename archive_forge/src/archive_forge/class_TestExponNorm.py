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
class TestExponNorm:

    def test_moments(self):

        def get_moms(lam, sig, mu):
            opK2 = 1.0 + 1 / (lam * sig) ** 2
            exp_skew = 2 / (lam * sig) ** 3 * opK2 ** (-1.5)
            exp_kurt = 6.0 * (1 + (lam * sig) ** 2) ** (-2)
            return [mu + 1 / lam, sig * sig + 1.0 / (lam * lam), exp_skew, exp_kurt]
        mu, sig, lam = (0, 1, 1)
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))
        mu, sig, lam = (-3, 2, 0.1)
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))
        mu, sig, lam = (0, 3, 1)
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))
        mu, sig, lam = (-5, 11, 3.5)
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))

    def test_nan_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.exponnorm.fit, x, floc=0, fscale=1)

    def test_inf_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.exponnorm.fit, x, floc=0, fscale=1)

    def test_extremes_x(self):
        assert_almost_equal(stats.exponnorm.pdf(-900, 1), 0.0)
        assert_almost_equal(stats.exponnorm.pdf(+900, 1), 0.0)
        assert_almost_equal(stats.exponnorm.pdf(-900, 0.01), 0.0)
        assert_almost_equal(stats.exponnorm.pdf(+900, 0.01), 0.0)

    @pytest.mark.parametrize('x, K, expected', [(20, 0.01, 6.90010764753618e-88), (1, 0.01, 0.24438994313247364), (-1, 0.01, 0.23955149623472075), (-20, 0.01, 4.6004708690125477e-88), (10, 1, 7.48518298877006e-05), (10, 10000, 9.990005048283775e-05)])
    def test_std_pdf(self, x, K, expected):
        assert_allclose(stats.exponnorm.pdf(x, K), expected, rtol=5e-12)

    @pytest.mark.parametrize('x, K, scale, expected', [[0, 0.01, 1, 0.4960109760186432], [-5, 0.005, 1, 2.7939945412195734e-07], [-10000.0, 0.01, 100, 0.0], [-10000.0, 0.01, 1000, 6.920401854427357e-24], [5, 0.001, 1, 0.9999997118542392]])
    def test_cdf_small_K(self, x, K, scale, expected):
        p = stats.exponnorm.cdf(x, K, scale=scale)
        if expected == 0.0:
            assert p == 0.0
        else:
            assert_allclose(p, expected, rtol=1e-13)

    @pytest.mark.parametrize('x, K, scale, expected', [[10, 0.01, 1, 8.474702916146657e-24], [2, 0.005, 1, 0.02302280664231312], [5, 0.005, 0.5, 8.024820681931086e-24], [10, 0.005, 0.5, 3.0603340062892486e-89], [20, 0.005, 0.5, 0.0], [-3, 0.001, 1, 0.9986545205566117]])
    def test_sf_small_K(self, x, K, scale, expected):
        p = stats.exponnorm.sf(x, K, scale=scale)
        if expected == 0.0:
            assert p == 0.0
        else:
            assert_allclose(p, expected, rtol=5e-13)