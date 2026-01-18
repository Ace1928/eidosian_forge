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
def test_skew(self):
    for n in self.get_n():
        x, y, xm, ym = self.generate_xy_sample(n)
        r = stats.skew(x)
        rm = stats.mstats.skew(xm)
        assert_almost_equal(r, rm, 10)
        r = stats.skew(y)
        rm = stats.mstats.skew(ym)
        assert_almost_equal(r, rm, 10)