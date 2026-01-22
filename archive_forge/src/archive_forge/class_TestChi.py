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
class TestChi:
    CHI_SF_10_4 = 9.83662422461598e-21
    CHI_MEAN_1000 = 31.61487189698

    def test_sf(self):
        s = stats.chi.sf(10, 4)
        assert_allclose(s, self.CHI_SF_10_4, rtol=1e-15)

    def test_isf(self):
        x = stats.chi.isf(self.CHI_SF_10_4, 4)
        assert_allclose(x, 10, rtol=1e-15)

    @pytest.mark.parametrize('df, ref', [(1000.0, CHI_MEAN_1000), (100000000000000.0, 9999999.999999976)])
    def test_mean(self, df, ref):
        assert_allclose(stats.chi.mean(df), ref, rtol=1e-12)

    @pytest.mark.parametrize('df, ref', [(0.0001, -9989.7316027504), (1, 0.7257913526447274), (1000.0, 1.0721981095025448), (10000000000.0, 1.0723649429080335), (1e+100, 1.0723649429247002)])
    def test_entropy(self, df, ref):
        assert_allclose(stats.chi(df).entropy(), ref, rtol=1e-15)