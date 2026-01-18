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
@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason='docstring stripped')
def test_signature_inspection(self):
    dummy_distr = _distr_gen(name='dummy')
    assert_equal(dummy_distr.numargs, 1)
    assert_equal(dummy_distr.shapes, 'a')
    res = re.findall('logpdf\\(x, a, loc=0, scale=1\\)', dummy_distr.__doc__)
    assert_(len(res) == 1)