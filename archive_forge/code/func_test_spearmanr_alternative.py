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
def test_spearmanr_alternative(self):
    x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1, 1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7]
    y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6, 0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4]
    r_exp = 0.6887298747763864
    r, p = mstats.spearmanr(x, y)
    assert_allclose(r, r_exp)
    assert_allclose(p, 0.004519192910756)
    r, p = mstats.spearmanr(x, y, alternative='greater')
    assert_allclose(r, r_exp)
    assert_allclose(p, 0.002259596455378)
    r, p = mstats.spearmanr(x, y, alternative='less')
    assert_allclose(r, r_exp)
    assert_allclose(p, 0.9977404035446)
    n = 100
    x = np.linspace(0, 5, n)
    y = 0.1 * x + np.random.rand(n)
    stat1, p1 = mstats.spearmanr(x, y)
    stat2, p2 = mstats.spearmanr(x, y, alternative='greater')
    assert_allclose(p2, p1 / 2)
    stat3, p3 = mstats.spearmanr(x, y, alternative='less')
    assert_allclose(p3, 1 - p1 / 2)
    assert stat1 == stat2 == stat3
    with pytest.raises(ValueError, match="alternative must be 'less'..."):
        mstats.spearmanr(x, y, alternative='ekki-ekki')