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
def test_reference_values(self):
    a = 1.0
    b = 3.0
    c = 2.0
    x_med = np.sqrt(1 - np.log(0.5 + np.exp(-(8.0 + np.log(2.0)))))
    cdf = stats.truncweibull_min.cdf(x_med, c, a, b)
    assert_allclose(cdf, 0.5)
    lc = stats.truncweibull_min.logcdf(x_med, c, a, b)
    assert_allclose(lc, -np.log(2.0))
    ppf = stats.truncweibull_min.ppf(0.5, c, a, b)
    assert_allclose(ppf, x_med)
    sf = stats.truncweibull_min.sf(x_med, c, a, b)
    assert_allclose(sf, 0.5)
    ls = stats.truncweibull_min.logsf(x_med, c, a, b)
    assert_allclose(ls, -np.log(2.0))
    isf = stats.truncweibull_min.isf(0.5, c, a, b)
    assert_allclose(isf, x_med)