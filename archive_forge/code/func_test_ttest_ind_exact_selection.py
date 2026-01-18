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
def test_ttest_ind_exact_selection(self):
    np.random.seed(0)
    N = 3
    a = np.random.rand(N)
    b = np.random.rand(N)
    res0 = stats.ttest_ind(a, b)
    res1 = stats.ttest_ind(a, b, permutations=1000)
    res2 = stats.ttest_ind(a, b, permutations=0)
    res3 = stats.ttest_ind(a, b, permutations=np.inf)
    assert res1.pvalue != res0.pvalue
    assert res2.pvalue == res0.pvalue
    assert res3.pvalue == res1.pvalue