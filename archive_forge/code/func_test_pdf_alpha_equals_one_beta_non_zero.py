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
@pytest.mark.parametrize('method,decimal_places', [['dni', 4], ['piecewise', 4]])
def test_pdf_alpha_equals_one_beta_non_zero(self, method, decimal_places):
    """ sample points extracted from Tables and Graphs of Stable
        Probability Density Functions - Donald R Holt - 1973 - p 187.
        """
    xs = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    density = np.array([0.3183, 0.3096, 0.2925, 0.2622, 0.1591, 0.1587, 0.1599, 0.1635, 0.0637, 0.0729, 0.0812, 0.0955, 0.0318, 0.039, 0.0458, 0.0586, 0.0187, 0.0236, 0.0285, 0.0384])
    betas = np.array([0, 0.25, 0.5, 1, 0, 0.25, 0.5, 1, 0, 0.25, 0.5, 1, 0, 0.25, 0.5, 1, 0, 0.25, 0.5, 1])
    with np.errstate(all='ignore'), suppress_warnings() as sup:
        sup.filter(category=RuntimeWarning, message='Density calculation unstable.*')
        stats.levy_stable.pdf_default_method = method
        pdf = stats.levy_stable.pdf(xs, 1, betas, scale=1, loc=0)
        assert_almost_equal(pdf, density, decimal_places, method)