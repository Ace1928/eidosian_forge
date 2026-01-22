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
class TestNormInvGauss:

    def setup_method(self):
        np.random.seed(1234)

    def test_cdf_R(self):
        r_cdf = np.array([8.034920282e-07, 2.512671945e-05, 0.3186661051, 0.9988650664, 0.9999848769])
        x_test = np.array([-7, -5, 0, 8, 15])
        vals_cdf = stats.norminvgauss.cdf(x_test, a=1, b=0.5)
        assert_allclose(vals_cdf, r_cdf, atol=1e-09)

    def test_pdf_R(self):
        r_pdf = np.array([1.359600783e-06, 4.413878805e-05, 0.4555014266, 0.0007450485342, 8.917889931e-06])
        x_test = np.array([-7, -5, 0, 8, 15])
        vals_pdf = stats.norminvgauss.pdf(x_test, a=1, b=0.5)
        assert_allclose(vals_pdf, r_pdf, atol=1e-09)

    @pytest.mark.parametrize('x, a, b, sf, rtol', [(-1, 1, 0, 0.8759652211005315, 1e-13), (25, 1, 0, 1.1318690184042579e-13, 0.0001), (1, 5, -1.5, 0.002066711134653577, 1e-12), (10, 5, -1.5, 2.308435233930669e-29, 1e-09)])
    def test_sf_isf_mpmath(self, x, a, b, sf, rtol):
        s = stats.norminvgauss.sf(x, a, b)
        assert_allclose(s, sf, rtol=rtol)
        i = stats.norminvgauss.isf(sf, a, b)
        assert_allclose(i, x, rtol=rtol)

    def test_sf_isf_mpmath_vectorized(self):
        x = [-1, 25]
        a = [1, 1]
        b = 0
        sf = [0.8759652211005315, 1.1318690184042579e-13]
        s = stats.norminvgauss.sf(x, a, b)
        assert_allclose(s, sf, rtol=1e-13, atol=1e-16)
        i = stats.norminvgauss.isf(sf, a, b)
        assert_allclose(i, x, rtol=1e-06)

    def test_gh8718(self):
        dst = stats.norminvgauss(1, 0)
        x = np.arange(0, 20, 2)
        sf = dst.sf(x)
        isf = dst.isf(sf)
        assert_allclose(isf, x)

    def test_stats(self):
        a, b = (1, 0.5)
        gamma = np.sqrt(a ** 2 - b ** 2)
        v_stats = (b / gamma, a ** 2 / gamma ** 3, 3.0 * b / (a * np.sqrt(gamma)), 3.0 * (1 + 4 * b ** 2 / a ** 2) / gamma)
        assert_equal(v_stats, stats.norminvgauss.stats(a, b, moments='mvsk'))

    def test_ppf(self):
        a, b = (1, 0.5)
        x_test = np.array([0.001, 0.5, 0.999])
        vals = stats.norminvgauss.ppf(x_test, a, b)
        assert_allclose(x_test, stats.norminvgauss.cdf(vals, a, b))