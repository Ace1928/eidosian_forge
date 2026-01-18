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
def test_runstest(reset_randomstate):
    x = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1])
    z_twosided = 1.38675
    pvalue_twosided = 0.1655179
    z_greater = 1.38675
    pvalue_greater = 0.08275893
    z_less = 1.38675
    pvalue_less = 0.917241
    assert_almost_equal(np.array(Runs(x).runs_test(correction=False)), [z_twosided, pvalue_twosided], decimal=6)
    assert_almost_equal(runstest_1samp(x, correction=False), [z_twosided, pvalue_twosided], decimal=6)
    x2 = x - 0.5 + np.random.uniform(-0.1, 0.1, size=len(x))
    assert_almost_equal(runstest_1samp(x2, cutoff=0, correction=False), [z_twosided, pvalue_twosided], decimal=6)
    assert_almost_equal(runstest_1samp(x2, cutoff='mean', correction=False), [z_twosided, pvalue_twosided], decimal=6)
    assert_almost_equal(runstest_1samp(x2, cutoff=x2.mean(), correction=False), [z_twosided, pvalue_twosided], decimal=6)
    assert_almost_equal(runstest_1samp(x2, cutoff='median', correction=False), runstest_1samp(x2, cutoff=np.median(x2), correction=False), decimal=6)