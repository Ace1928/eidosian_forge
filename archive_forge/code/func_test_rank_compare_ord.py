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
def test_rank_compare_ord():
    levels = [-2, -1, 0, 1, 2]
    new = [24, 37, 21, 19, 6]
    active = [11, 51, 22, 21, 7]
    x1 = np.repeat(levels, new)
    x2 = np.repeat(levels, active)
    for use_t in [False, True]:
        res2 = rank_compare_2indep(x1, x2, use_t=use_t)
        res1 = rank_compare_2ordinal(new, active, use_t=use_t)
        assert_allclose(res2.prob1, res1.prob1, rtol=1e-13)
        assert_allclose(res2.var_prob, res1.var_prob, rtol=1e-13)
        s1 = str(res1.summary())
        s2 = str(res2.summary())
        assert s1 == s2