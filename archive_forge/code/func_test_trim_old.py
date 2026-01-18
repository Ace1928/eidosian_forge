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
def test_trim_old(self):
    x = ma.arange(100)
    assert_equal(mstats.trimboth(x).count(), 60)
    assert_equal(mstats.trimtail(x, tail='r').count(), 80)
    x[50:70] = masked
    trimx = mstats.trimboth(x)
    assert_equal(trimx.count(), 48)
    assert_equal(trimx._mask, [1] * 16 + [0] * 34 + [1] * 20 + [0] * 14 + [1] * 16)
    x._mask = nomask
    x.shape = (10, 10)
    assert_equal(mstats.trimboth(x).count(), 60)
    assert_equal(mstats.trimtail(x).count(), 80)