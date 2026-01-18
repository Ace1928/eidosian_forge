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
def test_warnings_gh_14019(self):
    rng = np.random.default_rng(abs(hash('test_warnings_gh_14019')))
    data1 = rng.random(size=881) + 0.5
    data2 = rng.random(size=369)
    message = 'ks_2samp: Exact calculation unsuccessful'
    with pytest.warns(RuntimeWarning, match=message):
        res = stats.ks_2samp(data1, data2, alternative='less')
        assert_allclose(res.pvalue, 0, atol=1e-14)