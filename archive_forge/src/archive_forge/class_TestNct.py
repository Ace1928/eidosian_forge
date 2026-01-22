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
class TestNct:

    def test_nc_parameter(self):
        rv = stats.nct(5, 0)
        assert_equal(rv.cdf(0), 0.5)
        rv = stats.nct(5, -1)
        assert_almost_equal(rv.cdf(0), 0.841344746069, decimal=10)

    def test_broadcasting(self):
        res = stats.nct.pdf(5, np.arange(4, 7)[:, None], np.linspace(0.1, 1, 4))
        expected = array([[0.00321886, 0.00557466, 0.00918418, 0.01442997], [0.00217142, 0.00395366, 0.00683888, 0.01126276], [0.00153078, 0.00291093, 0.00525206, 0.00900815]])
        assert_allclose(res, expected, rtol=1e-05)

    def test_variance_gh_issue_2401(self):
        rv = stats.nct(4, 0)
        assert_equal(rv.var(), 2.0)

    def test_nct_inf_moments(self):
        m, v, s, k = stats.nct.stats(df=0.9, nc=0.3, moments='mvsk')
        assert_equal([m, v, s, k], [np.nan, np.nan, np.nan, np.nan])
        m, v, s, k = stats.nct.stats(df=1.9, nc=0.3, moments='mvsk')
        assert_(np.isfinite(m))
        assert_equal([v, s, k], [np.nan, np.nan, np.nan])
        m, v, s, k = stats.nct.stats(df=3.1, nc=0.3, moments='mvsk')
        assert_(np.isfinite([m, v, s]).all())
        assert_equal(k, np.nan)

    def test_nct_stats_large_df_values(self):
        nct_mean_df_1000 = stats.nct.mean(1000, 2)
        nct_stats_df_1000 = stats.nct.stats(1000, 2)
        expected_stats_df_1000 = [2.0015015641422464, 1.0040115288163005]
        assert_allclose(nct_mean_df_1000, expected_stats_df_1000[0], rtol=1e-10)
        assert_allclose(nct_stats_df_1000, expected_stats_df_1000, rtol=1e-10)
        nct_mean = stats.nct.mean(100000, 2)
        nct_stats = stats.nct.stats(100000, 2)
        expected_stats = [2.0000150001562518, 1.0000400011500288]
        assert_allclose(nct_mean, expected_stats[0], rtol=1e-10)
        assert_allclose(nct_stats, expected_stats, rtol=1e-09)

    def test_cdf_large_nc(self):
        assert_allclose(stats.nct.cdf(2, 2, float(2 ** 16)), 0)