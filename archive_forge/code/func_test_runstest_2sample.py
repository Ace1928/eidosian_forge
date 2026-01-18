from statsmodels.compat.python import lzip
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
from scipy import stats
import pytest
from statsmodels.stats.contingency_tables import (
from statsmodels.sandbox.stats.runs import (Runs,
from statsmodels.sandbox.stats.runs import mcnemar as sbmcnemar
from statsmodels.stats.nonparametric import (
from statsmodels.tools.testing import Holder
def test_runstest_2sample():
    x = [31.8, 32.8, 39.2, 36, 30, 34.5, 37.4]
    y = [35.5, 27.6, 21.3, 24.8, 36.7, 30]
    y[-1] += 1e-06
    groups = np.concatenate((np.zeros(len(x)), np.ones(len(y))))
    res = runstest_2samp(x, y)
    res1 = (0.022428065200812752, 0.9821064931864921)
    assert_allclose(res, res1, rtol=1e-06)
    res2 = runstest_2samp(x, y)
    assert_allclose(res2, res, rtol=1e-06)
    xy = np.concatenate((x, y))
    res_1s = runstest_1samp(xy)
    assert_allclose(res_1s, res1, rtol=1e-06)
    res2_1s = runstest_1samp(xy, xy.mean())
    assert_allclose(res2_1s, res_1s, rtol=1e-06)