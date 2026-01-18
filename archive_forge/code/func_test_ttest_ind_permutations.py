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
@pytest.mark.parametrize('a,b,update,p_d', params)
def test_ttest_ind_permutations(self, a, b, update, p_d):
    options_a = {'axis': None, 'equal_var': False}
    options_p = {'axis': None, 'equal_var': False, 'permutations': 1000, 'random_state': 0}
    options_a.update(update)
    options_p.update(update)
    stat_a, _ = stats.ttest_ind(a, b, **options_a)
    stat_p, pvalue = stats.ttest_ind(a, b, **options_p)
    assert_array_almost_equal(stat_a, stat_p, 5)
    assert_array_almost_equal(pvalue, p_d)