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
@pytest.mark.parametrize('fun, args', [(stats.wilcoxon, (x,)), (stats.ks_1samp, (x, stats.norm.cdf)), (stats.ks_2samp, (x, y)), (stats.kstest, (x, y))])
def test_rename_mode_method(fun, args):
    res = fun(*args, method='exact')
    res2 = fun(*args, mode='exact')
    assert_equal(res, res2)
    err = f'{fun.__name__}() got multiple values for argument'
    with pytest.raises(TypeError, match=re.escape(err)):
        fun(*args, method='exact', mode='exact')