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
class TestGenpareto:

    def test_ab(self):
        for c in [1.0, 0.0]:
            c = np.asarray(c)
            a, b = stats.genpareto._get_support(c)
            assert_equal(a, 0.0)
            assert_(np.isposinf(b))
        c = np.asarray(-2.0)
        a, b = stats.genpareto._get_support(c)
        assert_allclose([a, b], [0.0, 0.5])

    def test_c0(self):
        rv = stats.genpareto(c=0.0)
        x = np.linspace(0, 10.0, 30)
        assert_allclose(rv.pdf(x), stats.expon.pdf(x))
        assert_allclose(rv.cdf(x), stats.expon.cdf(x))
        assert_allclose(rv.sf(x), stats.expon.sf(x))
        q = np.linspace(0.0, 1.0, 10)
        assert_allclose(rv.ppf(q), stats.expon.ppf(q))

    def test_cm1(self):
        rv = stats.genpareto(c=-1.0)
        x = np.linspace(0, 10.0, 30)
        assert_allclose(rv.pdf(x), stats.uniform.pdf(x))
        assert_allclose(rv.cdf(x), stats.uniform.cdf(x))
        assert_allclose(rv.sf(x), stats.uniform.sf(x))
        q = np.linspace(0.0, 1.0, 10)
        assert_allclose(rv.ppf(q), stats.uniform.ppf(q))
        assert_allclose(rv.logpdf(1), 0)

    def test_x_inf(self):
        rv = stats.genpareto(c=0.1)
        assert_allclose([rv.pdf(np.inf), rv.cdf(np.inf)], [0.0, 1.0])
        assert_(np.isneginf(rv.logpdf(np.inf)))
        rv = stats.genpareto(c=0.0)
        assert_allclose([rv.pdf(np.inf), rv.cdf(np.inf)], [0.0, 1.0])
        assert_(np.isneginf(rv.logpdf(np.inf)))
        rv = stats.genpareto(c=-1.0)
        assert_allclose([rv.pdf(np.inf), rv.cdf(np.inf)], [0.0, 1.0])
        assert_(np.isneginf(rv.logpdf(np.inf)))

    def test_c_continuity(self):
        x = np.linspace(0, 10, 30)
        for c in [0, -1]:
            pdf0 = stats.genpareto.pdf(x, c)
            for dc in [1e-14, -1e-14]:
                pdfc = stats.genpareto.pdf(x, c + dc)
                assert_allclose(pdf0, pdfc, atol=1e-12)
            cdf0 = stats.genpareto.cdf(x, c)
            for dc in [1e-14, 1e-14]:
                cdfc = stats.genpareto.cdf(x, c + dc)
                assert_allclose(cdf0, cdfc, atol=1e-12)

    def test_c_continuity_ppf(self):
        q = np.r_[np.logspace(1e-12, 0.01, base=0.1), np.linspace(0.01, 1, 30, endpoint=False), 1.0 - np.logspace(1e-12, 0.01, base=0.1)]
        for c in [0.0, -1.0]:
            ppf0 = stats.genpareto.ppf(q, c)
            for dc in [1e-14, -1e-14]:
                ppfc = stats.genpareto.ppf(q, c + dc)
                assert_allclose(ppf0, ppfc, atol=1e-12)

    def test_c_continuity_isf(self):
        q = np.r_[np.logspace(1e-12, 0.01, base=0.1), np.linspace(0.01, 1, 30, endpoint=False), 1.0 - np.logspace(1e-12, 0.01, base=0.1)]
        for c in [0.0, -1.0]:
            isf0 = stats.genpareto.isf(q, c)
            for dc in [1e-14, -1e-14]:
                isfc = stats.genpareto.isf(q, c + dc)
                assert_allclose(isf0, isfc, atol=1e-12)

    def test_cdf_ppf_roundtrip(self):
        q = np.r_[np.logspace(1e-12, 0.01, base=0.1), np.linspace(0.01, 1, 30, endpoint=False), 1.0 - np.logspace(1e-12, 0.01, base=0.1)]
        for c in [1e-08, -1e-18, 1e-15, -1e-15]:
            assert_allclose(stats.genpareto.cdf(stats.genpareto.ppf(q, c), c), q, atol=1e-15)

    def test_logsf(self):
        logp = stats.genpareto.logsf(10000000000.0, 0.01, 0, 1)
        assert_allclose(logp, -1842.0680753952365)

    @pytest.mark.parametrize('c, expected_stats', [(0, [1, 1, 2, 6]), (1 / 4, [4 / 3, 32 / 9, 10 / np.sqrt(2), np.nan]), (1 / 9, [9 / 8, 81 / 64 * (9 / 7), 10 / 9 * np.sqrt(7), 754 / 45]), (-1, [1 / 2, 1 / 12, 0, -6 / 5])])
    def test_stats(self, c, expected_stats):
        result = stats.genpareto.stats(c, moments='mvsk')
        assert_allclose(result, expected_stats, rtol=1e-13, atol=1e-15)

    def test_var(self):
        v = stats.genpareto.var(1e-08)
        assert_allclose(v, 1.000000040000001, rtol=1e-13)