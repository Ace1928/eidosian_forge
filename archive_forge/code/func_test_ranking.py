import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
def test_ranking(self):
    x = ma.array([0, 1, 1, 1, 2, 3, 4, 5, 5, 6])
    assert_almost_equal(mstats.rankdata(x), [1, 3, 3, 3, 5, 6, 7, 8.5, 8.5, 10])
    x[[3, 4]] = masked
    assert_almost_equal(mstats.rankdata(x), [1, 2.5, 2.5, 0, 0, 4, 5, 6.5, 6.5, 8])
    assert_almost_equal(mstats.rankdata(x, use_missing=True), [1, 2.5, 2.5, 4.5, 4.5, 4, 5, 6.5, 6.5, 8])
    x = ma.array([0, 1, 5, 1, 2, 4, 3, 5, 1, 6])
    assert_almost_equal(mstats.rankdata(x), [1, 3, 8.5, 3, 5, 7, 6, 8.5, 3, 10])
    x = ma.array([[0, 1, 1, 1, 2], [3, 4, 5, 5, 6]])
    assert_almost_equal(mstats.rankdata(x), [[1, 3, 3, 3, 5], [6, 7, 8.5, 8.5, 10]])
    assert_almost_equal(mstats.rankdata(x, axis=1), [[1, 3, 3, 3, 5], [1, 2, 3.5, 3.5, 5]])
    assert_almost_equal(mstats.rankdata(x, axis=0), [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])