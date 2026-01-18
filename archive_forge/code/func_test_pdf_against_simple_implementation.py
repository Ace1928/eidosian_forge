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
@pytest.mark.parametrize('rho, Gamma, rtol', [(36.545206797050334, 2.4952, 5e-13), (38.55107913669065, 2.085, 5e-13), (96292.3076923077, 0.0013, 5e-10)])
def test_pdf_against_simple_implementation(self, rho, Gamma, rtol):

    def pdf(E, M, Gamma):
        gamma = np.sqrt(M ** 2 * (M ** 2 + Gamma ** 2))
        k = 2 * np.sqrt(2) * M * Gamma * gamma / (np.pi * np.sqrt(M ** 2 + gamma))
        return k / ((E ** 2 - M ** 2) ** 2 + M ** 2 * Gamma ** 2)
    p = np.linspace(0.05, 0.95, 10)
    x = stats.rel_breitwigner.ppf(p, rho, scale=Gamma)
    res = stats.rel_breitwigner.pdf(x, rho, scale=Gamma)
    ref = pdf(x, rho * Gamma, Gamma)
    assert_allclose(res, ref, rtol=rtol)