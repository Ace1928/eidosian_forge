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
def test_pointbiserial():
    x = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    y = [14.8, 13.8, 12.4, 10.1, 7.1, 6.1, 5.8, 4.6, 4.3, 3.5, 3.3, 3.2, 3.0, 2.8, 2.8, 2.5, 2.4, 2.3, 2.1, 1.7, 1.7, 1.5, 1.3, 1.3, 1.2, 1.2, 1.1, 0.8, 0.7, 0.6, 0.5, 0.2, 0.2, 0.1]
    assert_almost_equal(stats.pointbiserialr(x, y)[0], 0.36149, 5)
    attributes = ('correlation', 'pvalue')
    res = stats.pointbiserialr(x, y)
    check_named_results(res, attributes)
    assert_equal(res.correlation, res.statistic)