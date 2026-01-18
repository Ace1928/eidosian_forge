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
def test_simard_lecuyer_table1(self):
    ns = [10, 50, 100, 200, 500, 1000]
    ratios = np.array([1.0 / 4, 1.0 / 3, 1.0 / 2, 1, 2, 3])
    expected = np.array([[1.92155292e-08, 5.72933228e-05, 0.0215233226, 0.631566589, 0.997685592, 0.999999942], [2.28096224e-09, 1.99142563e-05, 0.0142617934, 0.595345542, 0.996177701, 0.999998662], [1.00201886e-09, 1.32673079e-05, 0.0124608594, 0.58616322, 0.995866877, 0.99999824], [4.93313022e-10, 9.52658029e-06, 0.0112123138, 0.579486872, 0.995661824, 0.999997964], [2.37049293e-10, 6.85002458e-06, 0.0101309221, 0.573427224, 0.995491207, 0.99999775], [1.56990874e-10, 5.71738276e-06, 0.0095972543, 0.570322692, 0.995409545, 0.999997657]])
    for idx, n in enumerate(ns):
        x = ratios * np.log(2) * np.sqrt(np.pi / 2 / n)
        vals_cdf = stats.kstwo.cdf(x, n)
        assert_allclose(vals_cdf, expected[idx], rtol=1e-05)