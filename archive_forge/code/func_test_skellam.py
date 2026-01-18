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
def test_skellam(self):
    p1, p2 = (18, 22)
    m1 = stats.skellam.expect(lambda x: x, args=(p1, p2))
    m2 = stats.skellam.expect(lambda x: x ** 2, args=(p1, p2))
    assert_allclose(m1, p1 - p2, atol=1e-12)
    assert_allclose(m2 - m1 ** 2, p1 + p2, atol=1e-12)