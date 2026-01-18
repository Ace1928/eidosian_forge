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
def test_support_broadcasting_gh13294_regression():
    a0, b0 = stats.norm.support([0, 0, 0, 1], [1, 1, 1, -1])
    ex_a0 = np.array([-np.inf, -np.inf, -np.inf, np.nan])
    ex_b0 = np.array([np.inf, np.inf, np.inf, np.nan])
    assert_equal(a0, ex_a0)
    assert_equal(b0, ex_b0)
    assert a0.shape == ex_a0.shape
    assert b0.shape == ex_b0.shape
    a1, b1 = stats.norm.support([], [])
    ex_a1, ex_b1 = (np.array([]), np.array([]))
    assert_equal(a1, ex_a1)
    assert_equal(b1, ex_b1)
    assert a1.shape == ex_a1.shape
    assert b1.shape == ex_b1.shape
    a2, b2 = stats.norm.support([0, 0, 0, 1], [-1])
    ex_a2 = np.array(4 * [np.nan])
    ex_b2 = np.array(4 * [np.nan])
    assert_equal(a2, ex_a2)
    assert_equal(b2, ex_b2)
    assert a2.shape == ex_a2.shape
    assert b2.shape == ex_b2.shape