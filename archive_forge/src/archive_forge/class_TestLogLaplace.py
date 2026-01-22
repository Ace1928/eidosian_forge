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
class TestLogLaplace:

    def test_sf(self):
        c = np.array([2.0, 3.0, 5.0])
        x = np.array([1e-05, 10000000000.0, 1000000000000000.0])
        ref = [0.99999999995, 5e-31, 5e-76]
        assert_allclose(stats.loglaplace.sf(x, c), ref, rtol=1e-15)

    def test_isf(self):
        c = 3.25
        q = [0.8, 0.1, 1e-10, 1e-20, 1e-40]
        ref = [0.7543222539245642, 1.6408455124660906, 964.4916294395846, 1151387.578354072, 1640845512466.0906]
        assert_allclose(stats.loglaplace.isf(q, c), ref, rtol=1e-14)