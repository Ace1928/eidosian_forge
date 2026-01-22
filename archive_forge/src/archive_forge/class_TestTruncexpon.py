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
class TestTruncexpon:

    def test_sf_isf(self):
        b = [20, 100]
        x = [19.999999, 99.999999]
        ref = [2.0611546593828472e-15, 3.7200778266671455e-50]
        assert_allclose(stats.truncexpon.sf(x, b), ref, rtol=1.5e-10)
        assert_allclose(stats.truncexpon.isf(ref, b), x, rtol=1e-12)