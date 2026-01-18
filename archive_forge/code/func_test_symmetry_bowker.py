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
def test_symmetry_bowker():
    table = np.array([0, 3, 4, 4, 2, 4, 1, 2, 4, 3, 5, 3, 0, 0, 2, 2, 3, 0, 0, 1, 5, 5, 5, 5, 5]).reshape(5, 5)
    res = SquareTable(table, shift_zeros=False).symmetry()
    mcnemar5_1 = dict(statistic=7.001587, pvalue=0.7252951, parameters=(10,), distr='chi2')
    assert_allclose([res.statistic, res.pvalue], [mcnemar5_1['statistic'], mcnemar5_1['pvalue']], rtol=1e-07)
    res = SquareTable(1 + table, shift_zeros=False).symmetry()
    mcnemar5_1b = dict(statistic=5.355988, pvalue=0.8661652, parameters=(10,), distr='chi2')
    assert_allclose([res.statistic, res.pvalue], [mcnemar5_1b['statistic'], mcnemar5_1b['pvalue']], rtol=1e-07)
    table = np.array([2, 2, 3, 6, 2, 3, 4, 3, 6, 6, 6, 7, 1, 9, 6, 7, 1, 1, 9, 8, 0, 1, 8, 9, 4]).reshape(5, 5)
    res = SquareTable(table, shift_zeros=False).symmetry()
    mcnemar5_2 = dict(statistic=18.76432, pvalue=0.04336035, parameters=(10,), distr='chi2')
    assert_allclose([res.statistic, res.pvalue], [mcnemar5_2['statistic'], mcnemar5_2['pvalue']], rtol=1.5e-07)
    res = SquareTable(1 + table, shift_zeros=False).symmetry()
    mcnemar5_2b = dict(statistic=14.55256, pvalue=0.1492461, parameters=(10,), distr='chi2')
    assert_allclose([res.statistic, res.pvalue], [mcnemar5_2b['statistic'], mcnemar5_2b['pvalue']], rtol=1e-07)