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
class TestJohnsonsu:
    cases = [(-500, 1, 1, 0.9999999982660072, 1e-08), (2000, 1, 1, 7.426351000595343e-21, 5e-14), (100000, 1, 1, 4.046923979269977e-40, 5e-14)]

    @pytest.mark.parametrize('case', cases)
    def test_sf_isf(self, case):
        x, a, b, sf, tol = case
        assert_allclose(stats.johnsonsu.sf(x, a, b), sf, rtol=5e-14)
        assert_allclose(stats.johnsonsu.isf(sf, a, b), x, rtol=tol)