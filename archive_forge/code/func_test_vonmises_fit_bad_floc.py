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
@pytest.mark.xslow
def test_vonmises_fit_bad_floc(self):
    data = [-0.92923506, -0.32498224, 0.13054989, -0.97252014, 2.79658071, -0.89110948, 1.22520295, 1.44398065, 2.49163859, 1.50315096, 3.05437696, -2.73126329, -3.06272048, 1.64647173, 1.94509247, -1.14328023, 0.8499056, 2.36714682, -1.6823179, -0.88359996]
    data = np.asarray(data)
    loc = -0.5 * np.pi
    kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data, floc=loc)
    assert kappa_fit == np.finfo(float).tiny
    _assert_less_or_close_loglike(stats.vonmises, data, stats.vonmises.nnlf, fscale=1, floc=loc)