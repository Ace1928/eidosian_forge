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
def test_pdf_large_x(self):
    logpdfvals = [[40, -1, -1604.8342333663986], [40, -1 / 2, -1004.1429467237419], [40, 0, -800.9189385332047], [40, 1 / 2, -800.2257913526447], [-40, -1 / 2, -800.2257913526447], [-2, 10000000.0, -200000000000019.97], [2, -10000000.0, -200000000000019.97]]
    for x, a, logpdfval in logpdfvals:
        logp = stats.skewnorm.logpdf(x, a)
        assert_allclose(logp, logpdfval, rtol=1e-08)