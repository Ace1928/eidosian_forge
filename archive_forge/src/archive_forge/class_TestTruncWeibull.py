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
class TestTruncWeibull:

    def test_pdf_bounds(self):
        y = stats.truncweibull_min.pdf([0.1, 2.0], 2.0, 0.11, 1.99)
        assert_equal(y, [0.0, 0.0])

    def test_logpdf(self):
        y = stats.truncweibull_min.logpdf(2.0, 1.0, 2.0, np.inf)
        assert_equal(y, 0.0)
        y = stats.truncweibull_min.logpdf(2.0, 1.0, 2.0, 4.0)
        assert_allclose(y, 0.14541345786885884)

    def test_ppf_bounds(self):
        y = stats.truncweibull_min.ppf([0.0, 1.0], 2.0, 0.1, 2.0)
        assert_equal(y, [0.1, 2.0])

    def test_cdf_to_ppf(self):
        q = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        x = stats.truncweibull_min.ppf(q, 2.0, 0.0, 3.0)
        q_out = stats.truncweibull_min.cdf(x, 2.0, 0.0, 3.0)
        assert_allclose(q, q_out)

    def test_sf_to_isf(self):
        q = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        x = stats.truncweibull_min.isf(q, 2.0, 0.0, 3.0)
        q_out = stats.truncweibull_min.sf(x, 2.0, 0.0, 3.0)
        assert_allclose(q, q_out)

    def test_munp(self):
        c = 2.0
        a = 1.0
        b = 3.0

        def xnpdf(x, n):
            return x ** n * stats.truncweibull_min.pdf(x, c, a, b)
        m0 = stats.truncweibull_min.moment(0, c, a, b)
        assert_equal(m0, 1.0)
        m1 = stats.truncweibull_min.moment(1, c, a, b)
        m1_expected, _ = quad(lambda x: xnpdf(x, 1), a, b)
        assert_allclose(m1, m1_expected)
        m2 = stats.truncweibull_min.moment(2, c, a, b)
        m2_expected, _ = quad(lambda x: xnpdf(x, 2), a, b)
        assert_allclose(m2, m2_expected)
        m3 = stats.truncweibull_min.moment(3, c, a, b)
        m3_expected, _ = quad(lambda x: xnpdf(x, 3), a, b)
        assert_allclose(m3, m3_expected)
        m4 = stats.truncweibull_min.moment(4, c, a, b)
        m4_expected, _ = quad(lambda x: xnpdf(x, 4), a, b)
        assert_allclose(m4, m4_expected)

    def test_reference_values(self):
        a = 1.0
        b = 3.0
        c = 2.0
        x_med = np.sqrt(1 - np.log(0.5 + np.exp(-(8.0 + np.log(2.0)))))
        cdf = stats.truncweibull_min.cdf(x_med, c, a, b)
        assert_allclose(cdf, 0.5)
        lc = stats.truncweibull_min.logcdf(x_med, c, a, b)
        assert_allclose(lc, -np.log(2.0))
        ppf = stats.truncweibull_min.ppf(0.5, c, a, b)
        assert_allclose(ppf, x_med)
        sf = stats.truncweibull_min.sf(x_med, c, a, b)
        assert_allclose(sf, 0.5)
        ls = stats.truncweibull_min.logsf(x_med, c, a, b)
        assert_allclose(ls, -np.log(2.0))
        isf = stats.truncweibull_min.isf(0.5, c, a, b)
        assert_allclose(isf, x_med)

    def test_compare_weibull_min(self):
        x = 1.5
        c = 2.0
        a = 0.0
        b = np.inf
        scale = 3.0
        p = stats.weibull_min.pdf(x, c, scale=scale)
        p_trunc = stats.truncweibull_min.pdf(x, c, a, b, scale=scale)
        assert_allclose(p, p_trunc)
        lp = stats.weibull_min.logpdf(x, c, scale=scale)
        lp_trunc = stats.truncweibull_min.logpdf(x, c, a, b, scale=scale)
        assert_allclose(lp, lp_trunc)
        cdf = stats.weibull_min.cdf(x, c, scale=scale)
        cdf_trunc = stats.truncweibull_min.cdf(x, c, a, b, scale=scale)
        assert_allclose(cdf, cdf_trunc)
        lc = stats.weibull_min.logcdf(x, c, scale=scale)
        lc_trunc = stats.truncweibull_min.logcdf(x, c, a, b, scale=scale)
        assert_allclose(lc, lc_trunc)
        s = stats.weibull_min.sf(x, c, scale=scale)
        s_trunc = stats.truncweibull_min.sf(x, c, a, b, scale=scale)
        assert_allclose(s, s_trunc)
        ls = stats.weibull_min.logsf(x, c, scale=scale)
        ls_trunc = stats.truncweibull_min.logsf(x, c, a, b, scale=scale)
        assert_allclose(ls, ls_trunc)
        s = stats.truncweibull_min.sf(30, 2, a, b, scale=3)
        assert_allclose(s, np.exp(-100))
        ls = stats.truncweibull_min.logsf(30, 2, a, b, scale=3)
        assert_allclose(ls, -100)

    def test_compare_weibull_min2(self):
        c, a, b = (2.5, 0.25, 1.25)
        x = np.linspace(a, b, 100)
        pdf1 = stats.truncweibull_min.pdf(x, c, a, b)
        cdf1 = stats.truncweibull_min.cdf(x, c, a, b)
        norm = stats.weibull_min.cdf(b, c) - stats.weibull_min.cdf(a, c)
        pdf2 = stats.weibull_min.pdf(x, c) / norm
        cdf2 = (stats.weibull_min.cdf(x, c) - stats.weibull_min.cdf(a, c)) / norm
        np.testing.assert_allclose(pdf1, pdf2)
        np.testing.assert_allclose(cdf1, cdf2)