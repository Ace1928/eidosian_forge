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
class TestPowerlaw:

    @pytest.mark.parametrize('x, a, sf', [(0.25, 2.0, 0.9375), (0.99609375, 1 / 256, 1.528855235208108e-05)])
    def test_sf(self, x, a, sf):
        assert_allclose(stats.powerlaw.sf(x, a), sf, rtol=1e-15)

    @pytest.fixture(scope='function')
    def rng(self):
        return np.random.default_rng(1234)

    @pytest.mark.parametrize('rvs_shape', [0.1, 0.5, 0.75, 1, 2])
    @pytest.mark.parametrize('rvs_loc', [-1, 0, 1])
    @pytest.mark.parametrize('rvs_scale', [0.1, 1, 5])
    @pytest.mark.parametrize('fix_shape, fix_loc, fix_scale', [p for p in product([True, False], repeat=3) if False in p])
    def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale, fix_shape, fix_loc, fix_scale, rng):
        data = stats.powerlaw.rvs(size=250, a=rvs_shape, loc=rvs_loc, scale=rvs_scale, random_state=rng)
        kwds = dict()
        if fix_shape:
            kwds['f0'] = rvs_shape
        if fix_loc:
            kwds['floc'] = np.nextafter(data.min(), -np.inf)
        if fix_scale:
            kwds['fscale'] = rvs_scale
        _assert_less_or_close_loglike(stats.powerlaw, data, **kwds)

    def test_problem_case(self):
        a = 2.500028626451306
        location = 0.0
        scale = 35.249023299873095
        data = stats.powerlaw.rvs(a=a, loc=location, scale=scale, size=100, random_state=np.random.default_rng(5))
        kwds = {'fscale': np.ptp(data) * 2}
        _assert_less_or_close_loglike(stats.powerlaw, data, **kwds)

    def test_fit_warnings(self):
        assert_fit_warnings(stats.powerlaw)
        msg = " Maximum likelihood estimation with 'powerlaw' requires"
        with assert_raises(FitDataError, match=msg):
            stats.powerlaw.fit([1, 2, 4], floc=0, fscale=3)
        msg = " Maximum likelihood estimation with 'powerlaw' requires"
        with assert_raises(FitDataError, match=msg):
            stats.powerlaw.fit([1, 2, 4], floc=2)
        msg = " Maximum likelihood estimation with 'powerlaw' requires"
        with assert_raises(FitDataError, match=msg):
            stats.powerlaw.fit([1, 2, 4], floc=1)
        msg = 'Negative or zero `fscale` is outside'
        with assert_raises(ValueError, match=msg):
            stats.powerlaw.fit([1, 2, 4], fscale=-3)
        msg = '`fscale` must be greater than the range of data.'
        with assert_raises(ValueError, match=msg):
            stats.powerlaw.fit([1, 2, 4], fscale=3)

    def test_minimum_data_zero_gh17801(self):
        data = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6]
        dist = stats.powerlaw
        with np.errstate(over='ignore'):
            _assert_less_or_close_loglike(dist, data)