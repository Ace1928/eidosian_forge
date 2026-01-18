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
@pytest.mark.parametrize('chi, expected, rtol', [(0.5, (0.5964284712757741, 0.052890651988588604), 1e-12), (0.101, (0.5893490968089076, 0.053017469847275685), 1e-11), (0.1, (0.5893431757009437, 0.05301755449499372), 1e-13), (0.01, (0.5890515677940915, 0.05302167905837031), 1e-13), (0.001, (0.5890486520005177, 0.053021719862088104), 1e-13), (0.0001, (0.5890486228426105, 0.0530217202700811), 1e-13), (1e-06, (0.5890486225481156, 0.05302172027420182), 1e-13), (1e-09, (0.5890486225480862, 0.05302172027420224), 1e-13)])
def test_stats_small_chi(self, chi, expected, rtol):
    val = stats.argus.stats(chi, moments='mv')
    assert_allclose(val, expected, rtol=rtol)