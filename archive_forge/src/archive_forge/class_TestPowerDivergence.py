import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
class TestPowerDivergence:

    def check_power_divergence(self, f_obs, f_exp, ddof, axis, lambda_, expected_stat):
        f_obs = np.asarray(f_obs)
        if axis is None:
            num_obs = f_obs.size
        else:
            b = np.broadcast(f_obs, f_exp)
            num_obs = b.shape[axis]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'Mean of empty slice')
            stat, p = stats.power_divergence(f_obs=f_obs, f_exp=f_exp, ddof=ddof, axis=axis, lambda_=lambda_)
            assert_allclose(stat, expected_stat)
            if lambda_ == 1 or lambda_ == 'pearson':
                stat, p = stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof, axis=axis)
                assert_allclose(stat, expected_stat)
        ddof = np.asarray(ddof)
        expected_p = stats.distributions.chi2.sf(expected_stat, num_obs - 1 - ddof)
        assert_allclose(p, expected_p)

    def test_basic(self):
        for case in power_div_1d_cases:
            self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, None, case.chi2)
            self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 'pearson', case.chi2)
            self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 1, case.chi2)
            self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 'log-likelihood', case.log)
            self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 'mod-log-likelihood', case.mod_log)
            self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 'cressie-read', case.cr)
            self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 2 / 3, case.cr)

    def test_basic_masked(self):
        for case in power_div_1d_cases:
            mobs = np.ma.array(case.f_obs)
            self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, None, case.chi2)
            self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 'pearson', case.chi2)
            self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 1, case.chi2)
            self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 'log-likelihood', case.log)
            self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 'mod-log-likelihood', case.mod_log)
            self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 'cressie-read', case.cr)
            self.check_power_divergence(mobs, case.f_exp, case.ddof, case.axis, 2 / 3, case.cr)

    def test_axis(self):
        case0 = power_div_1d_cases[0]
        case1 = power_div_1d_cases[1]
        f_obs = np.vstack((case0.f_obs, case1.f_obs))
        f_exp = np.vstack((np.ones_like(case0.f_obs) * np.mean(case0.f_obs), case1.f_exp))
        self.check_power_divergence(f_obs, f_exp, 0, 1, 'pearson', [case0.chi2, case1.chi2])
        self.check_power_divergence(f_obs, f_exp, 0, 1, 'log-likelihood', [case0.log, case1.log])
        self.check_power_divergence(f_obs, f_exp, 0, 1, 'mod-log-likelihood', [case0.mod_log, case1.mod_log])
        self.check_power_divergence(f_obs, f_exp, 0, 1, 'cressie-read', [case0.cr, case1.cr])
        self.check_power_divergence(np.array(case0.f_obs).reshape(2, 2), None, 0, None, 'pearson', case0.chi2)

    def test_ddof_broadcasting(self):
        case0 = power_div_1d_cases[0]
        case1 = power_div_1d_cases[1]
        f_obs = np.vstack((case0.f_obs, case1.f_obs)).T
        f_exp = np.vstack((np.ones_like(case0.f_obs) * np.mean(case0.f_obs), case1.f_exp)).T
        expected_chi2 = [case0.chi2, case1.chi2]
        ddof = np.array([[0], [1]])
        stat, p = stats.power_divergence(f_obs, f_exp, ddof=ddof)
        assert_allclose(stat, expected_chi2)
        stat0, p0 = stats.power_divergence(f_obs, f_exp, ddof=ddof[0, 0])
        stat1, p1 = stats.power_divergence(f_obs, f_exp, ddof=ddof[1, 0])
        assert_array_equal(p, np.vstack((p0, p1)))

    def test_empty_cases(self):
        with warnings.catch_warnings():
            for case in power_div_empty_cases:
                self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 'pearson', case.chi2)
                self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 'log-likelihood', case.log)
                self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 'mod-log-likelihood', case.mod_log)
                self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis, 'cressie-read', case.cr)

    def test_power_divergence_result_attributes(self):
        f_obs = power_div_1d_cases[0].f_obs
        f_exp = power_div_1d_cases[0].f_exp
        ddof = power_div_1d_cases[0].ddof
        axis = power_div_1d_cases[0].axis
        res = stats.power_divergence(f_obs=f_obs, f_exp=f_exp, ddof=ddof, axis=axis, lambda_='pearson')
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_power_divergence_gh_12282(self):
        f_obs = np.array([[10, 20], [30, 20]])
        f_exp = np.array([[5, 15], [35, 25]])
        with assert_raises(ValueError, match='For each axis slice...'):
            stats.power_divergence(f_obs=[10, 20], f_exp=[30, 60])
        with assert_raises(ValueError, match='For each axis slice...'):
            stats.power_divergence(f_obs=f_obs, f_exp=f_exp, axis=1)
        stat, pval = stats.power_divergence(f_obs=f_obs, f_exp=f_exp)
        assert_allclose(stat, [5.71428571, 2.66666667])
        assert_allclose(pval, [0.01682741, 0.10247043])