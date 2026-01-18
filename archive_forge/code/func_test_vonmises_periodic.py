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
@pytest.mark.parametrize('k', [0.1, 1, 101])
@pytest.mark.parametrize('x', [0, 1, np.pi, 10, 100])
def test_vonmises_periodic(self, k, x):

    def check_vonmises_pdf_periodic(k, L, s, x):
        vm = stats.vonmises(k, loc=L, scale=s)
        assert_almost_equal(vm.pdf(x), vm.pdf(x % (2 * np.pi * s)))

    def check_vonmises_cdf_periodic(k, L, s, x):
        vm = stats.vonmises(k, loc=L, scale=s)
        assert_almost_equal(vm.cdf(x) % 1, vm.cdf(x % (2 * np.pi * s)) % 1)
    check_vonmises_pdf_periodic(k, 0, 1, x)
    check_vonmises_pdf_periodic(k, 1, 1, x)
    check_vonmises_pdf_periodic(k, 0, 10, x)
    check_vonmises_cdf_periodic(k, 0, 1, x)
    check_vonmises_cdf_periodic(k, 1, 1, x)
    check_vonmises_cdf_periodic(k, 0, 10, x)