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
class TestPercentileOfScore:

    def f(self, *args, **kwargs):
        return stats.percentileofscore(*args, **kwargs)

    @pytest.mark.parametrize('kind, result', [('rank', 40), ('mean', 35), ('strict', 30), ('weak', 40)])
    def test_unique(self, kind, result):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert_equal(self.f(a, 4, kind=kind), result)

    @pytest.mark.parametrize('kind, result', [('rank', 45), ('mean', 40), ('strict', 30), ('weak', 50)])
    def test_multiple2(self, kind, result):
        a = [1, 2, 3, 4, 4, 5, 6, 7, 8, 9]
        assert_equal(self.f(a, 4, kind=kind), result)

    @pytest.mark.parametrize('kind, result', [('rank', 50), ('mean', 45), ('strict', 30), ('weak', 60)])
    def test_multiple3(self, kind, result):
        a = [1, 2, 3, 4, 4, 4, 5, 6, 7, 8]
        assert_equal(self.f(a, 4, kind=kind), result)

    @pytest.mark.parametrize('kind, result', [('rank', 30), ('mean', 30), ('strict', 30), ('weak', 30)])
    def test_missing(self, kind, result):
        a = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        assert_equal(self.f(a, 4, kind=kind), result)

    @pytest.mark.parametrize('kind, result', [('rank', 40), ('mean', 35), ('strict', 30), ('weak', 40)])
    def test_large_numbers(self, kind, result):
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert_equal(self.f(a, 40, kind=kind), result)

    @pytest.mark.parametrize('kind, result', [('rank', 50), ('mean', 45), ('strict', 30), ('weak', 60)])
    def test_large_numbers_multiple3(self, kind, result):
        a = [10, 20, 30, 40, 40, 40, 50, 60, 70, 80]
        assert_equal(self.f(a, 40, kind=kind), result)

    @pytest.mark.parametrize('kind, result', [('rank', 30), ('mean', 30), ('strict', 30), ('weak', 30)])
    def test_large_numbers_missing(self, kind, result):
        a = [10, 20, 30, 50, 60, 70, 80, 90, 100, 110]
        assert_equal(self.f(a, 40, kind=kind), result)

    @pytest.mark.parametrize('kind, result', [('rank', [0, 10, 100, 100]), ('mean', [0, 5, 95, 100]), ('strict', [0, 0, 90, 100]), ('weak', [0, 10, 100, 100])])
    def test_boundaries(self, kind, result):
        a = [10, 20, 30, 50, 60, 70, 80, 90, 100, 110]
        assert_equal(self.f(a, [0, 10, 110, 200], kind=kind), result)

    @pytest.mark.parametrize('kind, result', [('rank', [0, 10, 100]), ('mean', [0, 5, 95]), ('strict', [0, 0, 90]), ('weak', [0, 10, 100])])
    def test_inf(self, kind, result):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, +np.inf]
        assert_equal(self.f(a, [-np.inf, 1, +np.inf], kind=kind), result)
    cases = [('propagate', [], 1, np.nan), ('propagate', [np.nan], 1, np.nan), ('propagate', [np.nan], [0, 1, 2], [np.nan, np.nan, np.nan]), ('propagate', [1, 2], [1, 2, np.nan], [50, 100, np.nan]), ('omit', [1, 2, np.nan], [0, 1, 2], [0, 50, 100]), ('omit', [1, 2], [0, 1, np.nan], [0, 50, np.nan]), ('omit', [np.nan, np.nan], [0, 1, 2], [np.nan, np.nan, np.nan])]

    @pytest.mark.parametrize('policy, a, score, result', cases)
    def test_nans_ok(self, policy, a, score, result):
        assert_equal(self.f(a, score, nan_policy=policy), result)
    cases = [('raise', [1, 2, 3, np.nan], [1, 2, 3], 'The input contains nan values'), ('raise', [1, 2, 3], [1, 2, 3, np.nan], 'The input contains nan values')]

    @pytest.mark.parametrize('policy, a, score, message', cases)
    def test_nans_fail(self, policy, a, score, message):
        with assert_raises(ValueError, match=message):
            self.f(a, score, nan_policy=policy)

    @pytest.mark.parametrize('shape', [(6,), (2, 3), (2, 1, 3), (2, 1, 1, 3)])
    def test_nd(self, shape):
        a = np.array([0, 1, 2, 3, 4, 5])
        scores = a.reshape(shape)
        results = scores * 10
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert_equal(self.f(a, scores, kind='rank'), results)