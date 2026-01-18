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
def test_pdf_R(self):
    r_pdf = np.array([1.359600783e-06, 4.413878805e-05, 0.4555014266, 0.0007450485342, 8.917889931e-06])
    x_test = np.array([-7, -5, 0, 8, 15])
    vals_pdf = stats.norminvgauss.pdf(x_test, a=1, b=0.5)
    assert_allclose(vals_pdf, r_pdf, atol=1e-09)