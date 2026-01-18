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
def test_rice_zero_b(self):
    x = [0.2, 1.0, 5.0]
    assert_(np.isfinite(stats.rice.pdf(x, b=0.0)).all())
    assert_(np.isfinite(stats.rice.logpdf(x, b=0.0)).all())
    assert_(np.isfinite(stats.rice.cdf(x, b=0.0)).all())
    assert_(np.isfinite(stats.rice.logcdf(x, b=0.0)).all())
    q = [0.1, 0.1, 0.5, 0.9]
    assert_(np.isfinite(stats.rice.ppf(q, b=0.0)).all())
    mvsk = stats.rice.stats(0, moments='mvsk')
    assert_(np.isfinite(mvsk).all())
    b = 1e-08
    assert_allclose(stats.rice.pdf(x, 0), stats.rice.pdf(x, b), atol=b, rtol=0)