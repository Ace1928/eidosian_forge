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
class TestLaplace:

    @pytest.mark.parametrize('rvs_loc', [-5, 0, 1, 2])
    @pytest.mark.parametrize('rvs_scale', [1, 2, 3, 10])
    def test_fit(self, rvs_loc, rvs_scale):
        data = stats.laplace.rvs(size=100, loc=rvs_loc, scale=rvs_scale)
        loc_mle = np.median(data)
        scale_mle = np.sum(np.abs(data - loc_mle)) / len(data)
        loc, scale = stats.laplace.fit(data)
        assert_allclose(loc, loc_mle, atol=1e-15, rtol=1e-15)
        assert_allclose(scale, scale_mle, atol=1e-15, rtol=1e-15)
        loc, scale = stats.laplace.fit(data, floc=loc_mle)
        assert_allclose(scale, scale_mle, atol=1e-15, rtol=1e-15)
        loc, scale = stats.laplace.fit(data, fscale=scale_mle)
        assert_allclose(loc, loc_mle)
        loc = rvs_loc * 2
        scale_mle = np.sum(np.abs(data - loc)) / len(data)
        loc, scale = stats.laplace.fit(data, floc=loc)
        assert_equal(scale_mle, scale)
        loc, scale = stats.laplace.fit(data, fscale=scale_mle)
        assert_equal(loc_mle, loc)
        assert_raises(RuntimeError, stats.laplace.fit, data, floc=loc_mle, fscale=scale_mle)
        assert_raises(ValueError, stats.laplace.fit, [np.nan])
        assert_raises(ValueError, stats.laplace.fit, [np.inf])

    @pytest.mark.parametrize('rvs_loc,rvs_scale', [(-5, 10), (10, 5), (0.5, 0.2)])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale):
        data = stats.laplace.rvs(size=1000, loc=rvs_loc, scale=rvs_scale)

        def ll(loc, scale, data):
            return -1 * (-len(data) * np.log(2 * scale) - 1 / scale * np.sum(np.abs(data - loc)))
        loc, scale = stats.laplace.fit(data)
        loc_opt, scale_opt = super(type(stats.laplace), stats.laplace).fit(data)
        ll_mle = ll(loc, scale, data)
        ll_opt = ll(loc_opt, scale_opt, data)
        assert ll_mle < ll_opt or np.allclose(ll_mle, ll_opt, atol=1e-15, rtol=1e-15)

    def test_fit_simple_non_random_data(self):
        data = np.array([1.0, 1.0, 3.0, 5.0, 8.0, 14.0])
        loc, scale = stats.laplace.fit(data, floc=6)
        assert_allclose(scale, 4, atol=1e-15, rtol=1e-15)
        loc, scale = stats.laplace.fit(data, fscale=6)
        assert_allclose(loc, 4, atol=1e-15, rtol=1e-15)

    def test_sf_cdf_extremes(self):
        x = 1000
        p0 = stats.laplace.cdf(-x)
        assert p0 == 0.0
        p1 = stats.laplace.cdf(x)
        assert p1 == 1.0
        p0 = stats.laplace.sf(x)
        assert p0 == 0.0
        p1 = stats.laplace.sf(-x)
        assert p1 == 1.0

    def test_sf(self):
        x = 200
        p = stats.laplace.sf(x)
        assert_allclose(p, np.exp(-x) / 2, rtol=1e-13)

    def test_isf(self):
        p = 1e-25
        x = stats.laplace.isf(p)
        assert_allclose(x, -np.log(2 * p), rtol=1e-13)