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
def test_pdf_r(self):
    vals_R = np.array([2.94895678275316e-13, 1.75746848647696e-10, 9.48149804073045e-08, 4.17862521692026e-05, 0.0103947630463822, 0.240864958986839, 0.162833527161649, 0.0374609592899472, 0.00634894847327781, 0.000941920705790324])
    lmbda, alpha, beta = (2, 2, 1)
    mu, delta = (0.5, 1.5)
    args = (lmbda, alpha * delta, beta * delta)
    gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
    x = np.linspace(-10, 10, 10)
    assert_allclose(gh.pdf(x), vals_R, atol=0, rtol=1e-13)