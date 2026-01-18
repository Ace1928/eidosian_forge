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
def test_norm(self):
    v = stats.norm.expect(lambda x: (x - 5) * (x - 5), loc=5, scale=2)
    assert_almost_equal(v, 4, decimal=14)
    m = stats.norm.expect(lambda x: x, loc=5, scale=2)
    assert_almost_equal(m, 5, decimal=14)
    lb = stats.norm.ppf(0.05, loc=5, scale=2)
    ub = stats.norm.ppf(0.95, loc=5, scale=2)
    prob90 = stats.norm.expect(lambda x: 1, loc=5, scale=2, lb=lb, ub=ub)
    assert_almost_equal(prob90, 0.9, decimal=14)
    prob90c = stats.norm.expect(lambda x: 1, loc=5, scale=2, lb=lb, ub=ub, conditional=True)
    assert_almost_equal(prob90c, 1.0, decimal=14)