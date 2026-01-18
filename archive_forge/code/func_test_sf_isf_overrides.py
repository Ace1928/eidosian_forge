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
@pytest.mark.parametrize('case', [('kappa3', None, None, None, None), ('loglaplace', None, None, None, None), ('lognorm', None, None, None, None), ('lomax', None, None, None, None), ('pareto', None, None, None, None)])
def test_sf_isf_overrides(case):
    distname, lp1, lp2, atol, rtol = case
    lpm = np.log10(0.5)
    lp1 = lp1 or -290
    lp2 = lp2 or -14
    atol = atol or 0
    rtol = rtol or 1e-12
    dist = getattr(stats, distname)
    params = dict(distcont)[distname]
    dist_frozen = dist(*params)
    ref = np.logspace(lp1, lpm)
    res = dist_frozen.sf(dist_frozen.isf(ref))
    assert_allclose(res, ref, atol=atol, rtol=rtol)
    ref = 1 - np.logspace(lp2, lpm, 20)
    res = dist_frozen.sf(dist_frozen.isf(ref))
    assert_allclose(res, ref, atol=atol, rtol=rtol)