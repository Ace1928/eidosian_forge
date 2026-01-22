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
class TestLognorm:

    def test_pdf(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            pdf = stats.lognorm.pdf([0, 0.5, 1], 1)
            assert_array_almost_equal(pdf, [0.0, 0.62749608, 0.39894228])

    def test_logcdf(self):
        x2, mu, sigma = (201.68, 195, 0.149)
        assert_allclose(stats.lognorm.sf(x2 - mu, s=sigma), stats.norm.sf(np.log(x2 - mu) / sigma))
        assert_allclose(stats.lognorm.logsf(x2 - mu, s=sigma), stats.norm.logsf(np.log(x2 - mu) / sigma))

    @pytest.fixture(scope='function')
    def rng(self):
        return np.random.default_rng(1234)

    @pytest.mark.parametrize('rvs_shape', [0.1, 2])
    @pytest.mark.parametrize('rvs_loc', [-2, 0, 2])
    @pytest.mark.parametrize('rvs_scale', [0.2, 1, 5])
    @pytest.mark.parametrize('fix_shape, fix_loc, fix_scale', [e for e in product((False, True), repeat=3) if False in e])
    @np.errstate(invalid='ignore')
    def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale, fix_shape, fix_loc, fix_scale, rng):
        data = stats.lognorm.rvs(size=100, s=rvs_shape, scale=rvs_scale, loc=rvs_loc, random_state=rng)
        kwds = {}
        if fix_shape:
            kwds['f0'] = rvs_shape
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale
        _assert_less_or_close_loglike(stats.lognorm, data, **kwds)

    def test_isf(self):
        s = 0.954
        q = [0.1, 2e-10, 5e-20, 6e-40]
        ref = [3.3960065375794937, 390.07632793595974, 5830.5020828128445, 287872.84087457904]
        assert_allclose(stats.lognorm.isf(q, s), ref, rtol=1e-14)