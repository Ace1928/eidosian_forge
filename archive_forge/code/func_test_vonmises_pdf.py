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
@pytest.mark.parametrize('x, kappa, expected_pdf', [(0.1, 0.01, 0.16074242744907072), (0.1, 25.0, 1.7515464099118245), (0.1, 800, 0.2073272544458798), (2.0, 0.01, 0.15849003875385817), (2.0, 25.0, 8.356882934278192e-16), (2.0, 800, 0.0)])
def test_vonmises_pdf(self, x, kappa, expected_pdf):
    pdf = stats.vonmises.pdf(x, kappa)
    assert_allclose(pdf, expected_pdf, rtol=1e-15)