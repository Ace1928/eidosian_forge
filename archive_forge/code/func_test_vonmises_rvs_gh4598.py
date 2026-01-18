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
def test_vonmises_rvs_gh4598(self):
    seed = abs(hash('von_mises_rvs'))
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    rng3 = np.random.default_rng(seed)
    rvs1 = stats.vonmises(1, loc=0, scale=1).rvs(random_state=rng1)
    rvs2 = stats.vonmises(1, loc=2 * np.pi, scale=1).rvs(random_state=rng2)
    rvs3 = stats.vonmises(1, loc=0, scale=2 * np.pi / abs(rvs1) + 1).rvs(random_state=rng3)
    assert_allclose(rvs1, rvs2, atol=1e-15)
    assert_allclose(rvs1, rvs3, atol=1e-15)