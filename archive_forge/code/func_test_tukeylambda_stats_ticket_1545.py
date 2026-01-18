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
def test_tukeylambda_stats_ticket_1545():
    mv = stats.tukeylambda.stats(0, moments='mvsk')
    expected = [0, np.pi ** 2 / 3, 0, 1.2]
    assert_almost_equal(mv, expected, decimal=10)
    mv = stats.tukeylambda.stats(3.13, moments='mvsk')
    expected = [0, 0.02692208588614651, 0, -0.8980623862192241]
    assert_almost_equal(mv, expected, decimal=10)
    mv = stats.tukeylambda.stats(0.14, moments='mvsk')
    expected = [0, 2.1102970222145023, 0, -0.027083773532230196]
    assert_almost_equal(mv, expected, decimal=10)