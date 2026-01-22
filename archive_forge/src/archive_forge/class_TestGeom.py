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
class TestGeom:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.geom.rvs(0.75, size=(2, 50))
        assert_(numpy.all(vals >= 0))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.geom.rvs(0.75)
        assert_(isinstance(val, int))
        val = stats.geom(0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_rvs_9313(self):
        rng = np.random.default_rng(649496242618848)
        rvs = stats.geom.rvs(np.exp(-35), size=5, random_state=rng)
        assert rvs.dtype == np.int64
        assert np.all(rvs > np.iinfo(np.int32).max)

    def test_pmf(self):
        vals = stats.geom.pmf([1, 2, 3], 0.5)
        assert_array_almost_equal(vals, [0.5, 0.25, 0.125])

    def test_logpmf(self):
        vals1 = np.log(stats.geom.pmf([1, 2, 3], 0.5))
        vals2 = stats.geom.logpmf([1, 2, 3], 0.5)
        assert_allclose(vals1, vals2, rtol=1e-15, atol=0)
        val = stats.geom.logpmf(1, 1)
        assert_equal(val, 0.0)

    def test_cdf_sf(self):
        vals = stats.geom.cdf([1, 2, 3], 0.5)
        vals_sf = stats.geom.sf([1, 2, 3], 0.5)
        expected = array([0.5, 0.75, 0.875])
        assert_array_almost_equal(vals, expected)
        assert_array_almost_equal(vals_sf, 1 - expected)

    def test_logcdf_logsf(self):
        vals = stats.geom.logcdf([1, 2, 3], 0.5)
        vals_sf = stats.geom.logsf([1, 2, 3], 0.5)
        expected = array([0.5, 0.75, 0.875])
        assert_array_almost_equal(vals, np.log(expected))
        assert_array_almost_equal(vals_sf, np.log1p(-expected))

    def test_ppf(self):
        vals = stats.geom.ppf([0.5, 0.75, 0.875], 0.5)
        expected = array([1.0, 2.0, 3.0])
        assert_array_almost_equal(vals, expected)

    def test_ppf_underflow(self):
        assert_allclose(stats.geom.ppf(1e-20, 1e-20), 1.0, atol=1e-14)

    def test_entropy_gh18226(self):
        h = stats.geom(0.0146).entropy()
        assert_allclose(h, 5.219397961962308, rtol=1e-15)