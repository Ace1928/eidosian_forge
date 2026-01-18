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
def test_pdf_norminvgauss(self):
    alpha, beta, delta, mu = (np.linspace(1, 20, 10), np.linspace(0, 19, 10) * np.float_power(-1, range(10)), np.linspace(1, 1, 10), np.linspace(-100, 100, 10))
    lmbda = -0.5
    args = (lmbda, alpha * delta, beta * delta)
    gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
    x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]
    assert_allclose(gh.pdf(x), stats.norminvgauss.pdf(x, a=alpha, b=beta, loc=mu, scale=delta), atol=0, rtol=1e-13)