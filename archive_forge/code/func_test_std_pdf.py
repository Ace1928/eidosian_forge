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
@pytest.mark.parametrize('x, K, expected', [(20, 0.01, 6.90010764753618e-88), (1, 0.01, 0.24438994313247364), (-1, 0.01, 0.23955149623472075), (-20, 0.01, 4.6004708690125477e-88), (10, 1, 7.48518298877006e-05), (10, 10000, 9.990005048283775e-05)])
def test_std_pdf(self, x, K, expected):
    assert_allclose(stats.exponnorm.pdf(x, K), expected, rtol=5e-12)