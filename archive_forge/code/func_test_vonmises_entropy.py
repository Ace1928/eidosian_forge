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
@pytest.mark.parametrize('kappa, expected_entropy', [(1, 1.6274014590199897), (5, 0.6756431570114528), (100, -0.8811275441649473), (1000, -2.03468891852547), (2000, -2.3813876496587847)])
def test_vonmises_entropy(self, kappa, expected_entropy):
    entropy = stats.vonmises.entropy(kappa)
    assert_allclose(entropy, expected_entropy, rtol=1e-13)