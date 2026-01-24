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
class TestRayleigh:

    def setup_method(self):
        np.random.seed(989284321)

    def test_logpdf(self):
        y = stats.rayleigh.logpdf(50)
        assert_allclose(y, -1246.0879769945718)

    def test_logsf(self):
        y = stats.rayleigh.logsf(50)
        assert_allclose(y, -1250)

    @pytest.mark.parametrize('rvs_loc,rvs_scale', [(0.85373171, 0.86932204), (0.20558821, 0.61621008)])
    def test_fit(self, rvs_loc, rvs_scale):
        data = stats.rayleigh.rvs(size=250, loc=rvs_loc, scale=rvs_scale)

        def scale_mle(data, floc):
            return (np.sum((data - floc) ** 2) / (2 * len(data))) ** 0.5
        scale_expect = scale_mle(data, rvs_loc)
        loc, scale = stats.rayleigh.fit(data, floc=rvs_loc)
        assert_equal(loc, rvs_loc)
        assert_equal(scale, scale_expect)
        loc, scale = stats.rayleigh.fit(data, fscale=0.6)
        assert_equal(scale, 0.6)
        loc, scale = stats.rayleigh.fit(data)
        assert_equal(scale, scale_mle(data, loc))

    @pytest.mark.parametrize('rvs_loc,rvs_scale', [[0.74, 0.01], [0.08464463, 0.12069025]])
    def test_fit_comparison_super_method(self, rvs_loc, rvs_scale):
        data = stats.rayleigh.rvs(size=250, loc=rvs_loc, scale=rvs_scale)
        _assert_less_or_close_loglike(stats.rayleigh, data)

    def test_fit_warnings(self):
        assert_fit_warnings(stats.rayleigh)

    def test_fit_gh17088(self):
        rng = np.random.default_rng(456)
        loc, scale, size = (50, 600, 500)
        rvs = stats.rayleigh.rvs(loc, scale, size=size, random_state=rng)
        loc_fit, _ = stats.rayleigh.fit(rvs)
        assert loc_fit < np.min(rvs)
        loc_fit, scale_fit = stats.rayleigh.fit(rvs, fscale=scale)
        assert loc_fit < np.min(rvs)
        assert scale_fit == scale