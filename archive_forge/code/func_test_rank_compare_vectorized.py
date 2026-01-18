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
def test_rank_compare_vectorized():
    np.random.seed(987126)
    x1 = np.random.randint(0, 20, (50, 3))
    x2 = np.random.randint(5, 25, (50, 3))
    res = rank_compare_2indep(x1, x2)
    tst = res.test_prob_superior(0.5)
    tost = res.tost_prob_superior(0.4, 0.6)
    res.summary()
    for i in range(3):
        res_i = rank_compare_2indep(x1[:, i], x2[:, i])
        assert_allclose(res.statistic[i], res_i.statistic, rtol=1e-14)
        assert_allclose(res.pvalue[i], res_i.pvalue, rtol=1e-14)
        assert_allclose(res.prob1[i], res_i.prob1, rtol=1e-14)
        tst_i = res_i.test_prob_superior(0.5)
        assert_allclose(tst.statistic[i], tst_i.statistic, rtol=1e-14)
        assert_allclose(tst.pvalue[i], tst_i.pvalue, rtol=1e-14)
        tost_i = res_i.tost_prob_superior(0.4, 0.6)
        assert_allclose(tost.statistic[i], tost_i.statistic, rtol=1e-14)
        assert_allclose(tost.pvalue[i], tost_i.pvalue, rtol=1e-14)