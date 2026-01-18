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
@pytest.mark.slow
def test_some_code_paths(self):
    from scipy.stats._stats_py import _count_paths_outside_method, _compute_outer_prob_inside_method
    _compute_outer_prob_inside_method(1, 1, 1, 1)
    _count_paths_outside_method(1000, 1, 1, 1001)
    with np.errstate(invalid='raise'):
        assert_raises(FloatingPointError, _count_paths_outside_method, 1100, 1099, 1, 1)
        assert_raises(FloatingPointError, _count_paths_outside_method, 2000, 1000, 1, 1)