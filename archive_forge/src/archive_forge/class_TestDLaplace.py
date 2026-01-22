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
class TestDLaplace:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.dlaplace.rvs(1.5, size=(2, 50))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.dlaplace.rvs(1.5)
        assert_(isinstance(val, int))
        val = stats.dlaplace(1.5).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])
        assert_(stats.dlaplace.rvs(0.8) is not None)

    def test_stats(self):
        a = 1.0
        dl = stats.dlaplace(a)
        m, v, s, k = dl.stats('mvsk')
        N = 37
        xx = np.arange(-N, N + 1)
        pp = dl.pmf(xx)
        m2, m4 = (np.sum(pp * xx ** 2), np.sum(pp * xx ** 4))
        assert_equal((m, s), (0, 0))
        assert_allclose((v, k), (m2, m4 / m2 ** 2 - 3.0), atol=1e-14, rtol=1e-08)

    def test_stats2(self):
        a = np.log(2.0)
        dl = stats.dlaplace(a)
        m, v, s, k = dl.stats('mvsk')
        assert_equal((m, s), (0.0, 0.0))
        assert_allclose((v, k), (4.0, 3.25))