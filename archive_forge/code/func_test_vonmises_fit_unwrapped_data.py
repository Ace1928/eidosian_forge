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
@pytest.mark.parametrize('sign', [-1, 1])
def test_vonmises_fit_unwrapped_data(self, sign):
    rng = np.random.default_rng(6762668991392531563)
    data = stats.vonmises(loc=sign * 0.5 * np.pi, kappa=10).rvs(100000, random_state=rng)
    shifted_data = data + 4 * np.pi
    kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data)
    kappa_fit_shifted, loc_fit_shifted, _ = stats.vonmises.fit(shifted_data)
    assert_allclose(loc_fit, loc_fit_shifted)
    assert_allclose(kappa_fit, kappa_fit_shifted)
    assert scale_fit == 1
    assert -np.pi < loc_fit < np.pi