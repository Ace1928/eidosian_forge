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
class TestPowerLogNorm:

    @pytest.mark.parametrize('x, c, s, ref', [(100, 20, 1, 1.9057100820561928e-114), (0.001, 20, 1, 0.9999999999507617), (0.001, 0.02, 1, 0.9999999999999508), (1e+22, 0.02, 1, 6.50744044621611e-12)])
    def test_sf(self, x, c, s, ref):
        assert_allclose(stats.powerlognorm.sf(x, c, s), ref, rtol=1e-13)

    @pytest.mark.parametrize('q, c, s, ref', [(0.9999999587870905, 0.02, 1, 0.01), (6.690376686108851e-233, 20, 1, 1000)])
    def test_isf(self, q, c, s, ref):
        assert_allclose(stats.powerlognorm.isf(q, c, s), ref, rtol=5e-11)

    @pytest.mark.parametrize('x, c, s, ref', [(1e+25, 0.02, 1, 0.9999999999999963), (1e-06, 0.02, 1, 2.054921078040843e-45), (1e-06, 200, 1, 2.0549210780408428e-41), (0.3, 200, 1, 0.9999999999713368)])
    def test_cdf(self, x, c, s, ref):
        assert_allclose(stats.powerlognorm.cdf(x, c, s), ref, rtol=3e-14)

    @pytest.mark.parametrize('x, c, s, ref', [(1e+22, 0.02, 1, 6.5954987852335016e-34), (1e+20, 0.001, 1, 1.588073750563988e-22), (1e+40, 0.001, 1, 1.3179391812506349e-43)])
    def test_pdf(self, x, c, s, ref):
        assert_allclose(stats.powerlognorm.pdf(x, c, s), ref, rtol=3e-12)