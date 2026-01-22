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
class TestRandInt:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.randint.rvs(5, 30, size=100)
        assert_(numpy.all(vals < 30) & numpy.all(vals >= 5))
        assert_(len(vals) == 100)
        vals = stats.randint.rvs(5, 30, size=(2, 50))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.randint.rvs(15, 46)
        assert_((val >= 15) & (val < 46))
        assert_(isinstance(val, numpy.ScalarType), msg=repr(type(val)))
        val = stats.randint(15, 46).rvs(3)
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_pdf(self):
        k = numpy.r_[0:36]
        out = numpy.where((k >= 5) & (k < 30), 1.0 / (30 - 5), 0)
        vals = stats.randint.pmf(k, 5, 30)
        assert_array_almost_equal(vals, out)

    def test_cdf(self):
        x = np.linspace(0, 36, 100)
        k = numpy.floor(x)
        out = numpy.select([k >= 30, k >= 5], [1.0, (k - 5.0 + 1) / (30 - 5.0)], 0)
        vals = stats.randint.cdf(x, 5, 30)
        assert_array_almost_equal(vals, out, decimal=12)