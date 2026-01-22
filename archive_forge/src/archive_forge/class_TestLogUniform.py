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
class TestLogUniform:

    def test_alias(self):
        rng = np.random.default_rng(98643218961)
        rv = stats.loguniform(10 ** (-3), 10 ** 0)
        rvs = rv.rvs(size=10000, random_state=rng)
        rng = np.random.default_rng(98643218961)
        rv2 = stats.reciprocal(10 ** (-3), 10 ** 0)
        rvs2 = rv2.rvs(size=10000, random_state=rng)
        assert_allclose(rvs2, rvs)
        vals, _ = np.histogram(np.log10(rvs), bins=10)
        assert 900 <= vals.min() <= vals.max() <= 1100
        assert np.abs(np.median(vals) - 1000) <= 10

    @pytest.mark.parametrize('method', ['mle', 'mm'])
    def test_fit_override(self, method):
        rng = np.random.default_rng(98643218961)
        rvs = stats.loguniform.rvs(0.1, 1, size=1000, random_state=rng)
        a, b, loc, scale = stats.loguniform.fit(rvs, method=method)
        assert scale == 1
        a, b, loc, scale = stats.loguniform.fit(rvs, fscale=2, method=method)
        assert scale == 2

    def test_overflow(self):
        rng = np.random.default_rng(7136519550773909093)
        a, b = (1e-200, 1e+200)
        dist = stats.loguniform(a, b)
        cdf = rng.uniform(0, 1, size=1000)
        assert_allclose(dist.cdf(dist.ppf(cdf)), cdf)
        rvs = dist.rvs(size=1000)
        assert_allclose(dist.ppf(dist.cdf(rvs)), rvs)
        x = 10.0 ** np.arange(-200, 200)
        pdf = dist.pdf(x)
        assert_allclose(pdf[:-1] / pdf[1:], 10)
        mean = (b - a) / (np.log(b) - np.log(a))
        assert_allclose(dist.mean(), mean)