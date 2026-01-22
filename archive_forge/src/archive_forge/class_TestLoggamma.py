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
class TestLoggamma:

    @pytest.mark.parametrize('x, c, cdf', [(1, 2, 0.7546378854206702), (-1, 14, 6.768116452566383e-18), (-745.1, 0.001, 0.4749605142005238), (-800, 0.001, 0.44958802911019136), (-725, 0.1, 3.4301205868273265e-32), (-740, 0.75, 1.0074360436599631e-241)])
    def test_cdf_ppf(self, x, c, cdf):
        p = stats.loggamma.cdf(x, c)
        assert_allclose(p, cdf, rtol=1e-13)
        y = stats.loggamma.ppf(cdf, c)
        assert_allclose(y, x, rtol=1e-13)

    @pytest.mark.parametrize('x, c, sf', [(4, 1.5, 1.6341528919488565e-23), (6, 100, 8.23836829202024e-74), (-800, 0.001, 0.5504119708898086), (-743, 0.0025, 0.8437131370024089)])
    def test_sf_isf(self, x, c, sf):
        s = stats.loggamma.sf(x, c)
        assert_allclose(s, sf, rtol=1e-13)
        y = stats.loggamma.isf(sf, c)
        assert_allclose(y, x, rtol=1e-13)

    def test_logpdf(self):
        lp = stats.loggamma.logpdf(-500, 2)
        assert_allclose(lp, -1000.0, rtol=1e-14)

    def test_stats(self):
        table = np.array([0.5, -1.9635, 4.9348, -1.5351, 4.0, 1.0, -0.5772, 1.6449, -1.1395, 2.4, 12.0, 2.4427, 0.0869, -0.2946, 0.1735]).reshape(-1, 5)
        for c, mean, var, skew, kurt in table:
            computed = stats.loggamma.stats(c, moments='msvk')
            assert_array_almost_equal(computed, [mean, var, skew, kurt], decimal=4)

    @pytest.mark.parametrize('c', [0.1, 0.001])
    def test_rvs(self, c):
        x = stats.loggamma.rvs(c, size=100000)
        assert np.isfinite(x).all()
        med = stats.loggamma.median(c)
        btest = stats.binomtest(np.count_nonzero(x < med), len(x))
        ci = btest.proportion_ci(confidence_level=0.999)
        assert ci.low < 0.5 < ci.high

    @pytest.mark.parametrize('c, ref', [(1e-08, 19.420680753952364), (1, 1.5772156649015328), (10000.0, -3.186214986116763), (10000000000.0, -10.093986931748889), (1e+100, -113.71031611649761)])
    def test_entropy(self, c, ref):
        assert_allclose(stats.loggamma.entropy(c), ref, rtol=1e-14)