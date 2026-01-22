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
class TestPlanck:

    def setup_method(self):
        np.random.seed(1234)

    def test_sf(self):
        vals = stats.planck.sf([1, 2, 3], 5.0)
        expected = array([4.5399929762484854e-05, 3.059023205018258e-07, 2.061153622438558e-09])
        assert_array_almost_equal(vals, expected)

    def test_logsf(self):
        vals = stats.planck.logsf([1000.0, 2000.0, 3000.0], 1000.0)
        expected = array([-1001000.0, -2001000.0, -3001000.0])
        assert_array_almost_equal(vals, expected)