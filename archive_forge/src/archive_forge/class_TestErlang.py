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
class TestErlang:

    def setup_method(self):
        np.random.seed(1234)

    def test_erlang_runtimewarning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            assert_raises(RuntimeWarning, stats.erlang.rvs, 1.3, loc=0, scale=1, size=4)
            data = [0.5, 1.0, 2.0, 4.0]
            result_erlang = stats.erlang.fit(data, f0=1)
            result_gamma = stats.gamma.fit(data, f0=1)
            assert_allclose(result_erlang, result_gamma, rtol=0.001)

    def test_gh_pr_10949_argcheck(self):
        assert_equal(stats.erlang.pdf(0.5, a=[1, -1]), stats.gamma.pdf(0.5, a=[1, -1]))