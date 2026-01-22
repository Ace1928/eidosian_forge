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
class TestNorm:

    def test_nan_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.norm.fit, x)

    def test_inf_raises_error(self):
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.norm.fit, x)

    def test_bad_keyword_arg(self):
        x = [1, 2, 3]
        assert_raises(TypeError, stats.norm.fit, x, plate='shrimp')

    @pytest.mark.parametrize('loc', [0, 1])
    def test_delta_cdf(self, loc):
        expected = 1.910641809677555e-28
        delta = stats.norm._delta_cdf(11 + loc, 12 + loc, loc=loc)
        assert_allclose(delta, expected, rtol=1e-13)
        delta = stats.norm._delta_cdf(-(12 + loc), -(11 + loc), loc=-loc)
        assert_allclose(delta, expected, rtol=1e-13)