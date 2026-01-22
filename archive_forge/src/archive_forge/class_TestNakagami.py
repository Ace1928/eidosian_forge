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
class TestNakagami:

    def test_logpdf(self):
        nu = 2.5
        x = 25
        logp = stats.nakagami.logpdf(x, nu)
        assert_allclose(logp, -1546.9253055607549)

    def test_sf_isf(self):
        nu = 2.5
        x0 = 5.0
        sf = stats.nakagami.sf(x0, nu)
        assert_allclose(sf, 2.736273158588307e-25, rtol=1e-13)
        x1 = stats.nakagami.isf(sf, nu)
        assert_allclose(x1, x0, rtol=1e-13)

    @pytest.mark.parametrize('m, ref', [(5, -0.097341814372152), (0.5, 0.7257913526447274), (10, -0.43426184310934907)])
    def test_entropy(self, m, ref):
        assert_allclose(stats.nakagami.entropy(m), ref, rtol=1.1e-14)

    @pytest.mark.parametrize('m, ref', [(1e-100, -5e+99), (1e-10, -4999999965.442979), (9999000.0, -7.333206478668433), (10010000.0, -7.3337562313259825), (10000000000.0, -10.787134112333835), (1e+100, -114.40346329705756)])
    def test_extreme_nu(self, m, ref):
        assert_allclose(stats.nakagami.entropy(m), ref)

    def test_entropy_overflow(self):
        assert np.isfinite(stats.nakagami._entropy(1e+100))
        assert np.isfinite(stats.nakagami._entropy(1e-100))

    @pytest.mark.parametrize('nu, ref', [(10000000000.0, 0.9999999999875), (1000.0, 0.9998750078173821), (1e-10, 1.772453850659802e-05)])
    def test_mean(self, nu, ref):
        assert_allclose(stats.nakagami.mean(nu), ref, rtol=1e-12)

    @pytest.mark.xfail(reason='Fit of nakagami not reliable, see gh-10908.')
    @pytest.mark.parametrize('nu', [1.6, 2.5, 3.9])
    @pytest.mark.parametrize('loc', [25.0, 10, 35])
    @pytest.mark.parametrize('scale', [13, 5, 20])
    def test_fit(self, nu, loc, scale):
        N = 100
        samples = stats.nakagami.rvs(size=N, nu=nu, loc=loc, scale=scale, random_state=1337)
        nu_est, loc_est, scale_est = stats.nakagami.fit(samples)
        assert_allclose(nu_est, nu, rtol=0.2)
        assert_allclose(loc_est, loc, rtol=0.2)
        assert_allclose(scale_est, scale, rtol=0.2)

        def dlogl_dnu(nu, loc, scale):
            return (-2 * nu + 1) * np.sum(1 / (samples - loc)) + 2 * nu / scale ** 2 * np.sum(samples - loc)

        def dlogl_dloc(nu, loc, scale):
            return N * (1 + np.log(nu) - polygamma(0, nu)) + 2 * np.sum(np.log((samples - loc) / scale)) - np.sum(((samples - loc) / scale) ** 2)

        def dlogl_dscale(nu, loc, scale):
            return -2 * N * nu / scale + 2 * nu / scale ** 3 * np.sum((samples - loc) ** 2)
        assert_allclose(dlogl_dnu(nu_est, loc_est, scale_est), 0, atol=0.001)
        assert_allclose(dlogl_dloc(nu_est, loc_est, scale_est), 0, atol=0.001)
        assert_allclose(dlogl_dscale(nu_est, loc_est, scale_est), 0, atol=0.001)

    @pytest.mark.parametrize('loc', [25.0, 10, 35])
    @pytest.mark.parametrize('scale', [13, 5, 20])
    def test_fit_nu(self, loc, scale):
        nu = 0.5
        n = 100
        samples = stats.nakagami.rvs(size=n, nu=nu, loc=loc, scale=scale, random_state=1337)
        nu_est, loc_est, scale_est = stats.nakagami.fit(samples, f0=nu)
        loc_theo = np.min(samples)
        scale_theo = np.sqrt(np.mean((samples - loc_est) ** 2))
        assert_allclose(nu_est, nu, rtol=1e-07)
        assert_allclose(loc_est, loc_theo, rtol=1e-07)
        assert_allclose(scale_est, scale_theo, rtol=1e-07)