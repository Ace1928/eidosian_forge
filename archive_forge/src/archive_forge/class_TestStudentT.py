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
class TestStudentT:

    def test_rvgeneric_std(self):
        assert_array_almost_equal(stats.t.std([5, 6]), [1.29099445, 1.22474487])

    def test_moments_t(self):
        assert_equal(stats.t.stats(df=1, moments='mvsk'), (np.inf, np.nan, np.nan, np.nan))
        assert_equal(stats.t.stats(df=1.01, moments='mvsk'), (0.0, np.inf, np.nan, np.nan))
        assert_equal(stats.t.stats(df=2, moments='mvsk'), (0.0, np.inf, np.nan, np.nan))
        assert_equal(stats.t.stats(df=2.01, moments='mvsk'), (0.0, 2.01 / (2.01 - 2.0), np.nan, np.inf))
        assert_equal(stats.t.stats(df=3, moments='sk'), (np.nan, np.inf))
        assert_equal(stats.t.stats(df=3.01, moments='sk'), (0.0, np.inf))
        assert_equal(stats.t.stats(df=4, moments='sk'), (0.0, np.inf))
        assert_equal(stats.t.stats(df=4.01, moments='sk'), (0.0, 6.0 / (4.01 - 4.0)))

    def test_t_entropy(self):
        df = [1, 2, 25, 100]
        expected = [2.5310242469692907, 1.9602792291600821, 1.459327578078393, 1.4289633653182439]
        assert_allclose(stats.t.entropy(df), expected, rtol=1e-13)

    @pytest.mark.parametrize('v, ref', [(100, 1.4289633653182439), (1e+100, 1.4189385332046727)])
    def test_t_extreme_entropy(self, v, ref):
        assert_allclose(stats.t.entropy(v), ref, rtol=1e-14)

    @pytest.mark.parametrize('methname', ['pdf', 'logpdf', 'cdf', 'ppf', 'sf', 'isf'])
    @pytest.mark.parametrize('df_infmask', [[0, 0], [1, 1], [0, 1], [[0, 1, 0], [1, 1, 1]], [[1, 0], [0, 1]], [[0], [1]]])
    def test_t_inf_df(self, methname, df_infmask):
        np.random.seed(0)
        df_infmask = np.asarray(df_infmask, dtype=bool)
        df = np.random.uniform(0, 10, size=df_infmask.shape)
        x = np.random.randn(*df_infmask.shape)
        df[df_infmask] = np.inf
        t_dist = stats.t(df=df, loc=3, scale=1)
        t_dist_ref = stats.t(df=df[~df_infmask], loc=3, scale=1)
        norm_dist = stats.norm(loc=3, scale=1)
        t_meth = getattr(t_dist, methname)
        t_meth_ref = getattr(t_dist_ref, methname)
        norm_meth = getattr(norm_dist, methname)
        res = t_meth(x)
        assert_equal(res[df_infmask], norm_meth(x[df_infmask]))
        assert_equal(res[~df_infmask], t_meth_ref(x[~df_infmask]))

    @pytest.mark.parametrize('df_infmask', [[0, 0], [1, 1], [0, 1], [[0, 1, 0], [1, 1, 1]], [[1, 0], [0, 1]], [[0], [1]]])
    def test_t_inf_df_stats_entropy(self, df_infmask):
        np.random.seed(0)
        df_infmask = np.asarray(df_infmask, dtype=bool)
        df = np.random.uniform(0, 10, size=df_infmask.shape)
        df[df_infmask] = np.inf
        res = stats.t.stats(df=df, loc=3, scale=1, moments='mvsk')
        res_ex_inf = stats.norm.stats(loc=3, scale=1, moments='mvsk')
        res_ex_noinf = stats.t.stats(df=df[~df_infmask], loc=3, scale=1, moments='mvsk')
        for i in range(4):
            assert_equal(res[i][df_infmask], res_ex_inf[i])
            assert_equal(res[i][~df_infmask], res_ex_noinf[i])
        res = stats.t.entropy(df=df, loc=3, scale=1)
        res_ex_inf = stats.norm.entropy(loc=3, scale=1)
        res_ex_noinf = stats.t.entropy(df=df[~df_infmask], loc=3, scale=1)
        assert_equal(res[df_infmask], res_ex_inf)
        assert_equal(res[~df_infmask], res_ex_noinf)

    def test_logpdf_pdf(self):
        x = [1, 1000.0, 10, 1]
        df = [1e+100, 1e+50, 1e+20, 1]
        logpdf_ref = [-1.4189385332046727, -500000.9189385332, -50.918938533204674, -1.8378770664093456]
        pdf_ref = [0.24197072451914334, 0, 7.69459862670642e-23, 0.15915494309189535]
        assert_allclose(stats.t.logpdf(x, df), logpdf_ref, rtol=1e-15)
        assert_allclose(stats.t.pdf(x, df), pdf_ref, rtol=1e-14)