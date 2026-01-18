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
@pytest.mark.parametrize('x, kappa, expected_logpdf', [(0.1, 0.01, -1.827952024600317), (0.1, 25.0, 0.5604990605420549), (0.1, 800, -1.5734567947337514), (2.0, 0.01, -1.8420635346185685), (2.0, 25.0, -34.718275985087146), (2.0, 800, -1130.4942582548683)])
def test_vonmises_logpdf(self, x, kappa, expected_logpdf):
    logpdf = stats.vonmises.logpdf(x, kappa)
    assert_allclose(logpdf, expected_logpdf, rtol=1e-15)