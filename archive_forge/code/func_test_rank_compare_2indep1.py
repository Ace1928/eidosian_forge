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
def test_rank_compare_2indep1():
    levels = [-2, -1, 0, 1, 2]
    new = [24, 37, 21, 19, 6]
    active = [11, 51, 22, 21, 7]
    x1 = np.repeat(levels, new)
    x2 = np.repeat(levels, active)
    res2_t = Holder(statistic=1.1757561456582, df=204.2984239868, pvalue=0.2410606649547, ci=[0.4700629827705593, 0.6183882855872511], prob=0.5442256341789052)
    res = rank_compare_2indep(x1, x2, use_t=False)
    assert_allclose(res.statistic, -res2_t.statistic, rtol=1e-13)
    assert_allclose(res.prob1, 1 - res2_t.prob, rtol=1e-13)
    assert_allclose(res.prob2, res2_t.prob, rtol=1e-13)
    tt = res.test_prob_superior()
    assert_allclose(tt[0], -res2_t.statistic, rtol=1e-13)
    ci = res.conf_int(alpha=0.05)
    assert_allclose(ci, 1 - np.array(res2_t.ci)[::-1], rtol=0.005)
    res_lb = res.test_prob_superior(value=ci[0])
    assert_allclose(res_lb[1], 0.05, rtol=1e-13)
    res_ub = res.test_prob_superior(value=ci[1])
    assert_allclose(res_ub[1], 0.05, rtol=1e-13)
    res_tost = res.tost_prob_superior(ci[0], ci[1] * 1.05)
    assert_allclose(res_tost.results_larger.pvalue, 0.025, rtol=1e-13)
    assert_allclose(res_tost.pvalue, 0.025, rtol=1e-13)
    res_tost = res.tost_prob_superior(ci[0] * 0.85, ci[1])
    assert_allclose(res_tost.results_smaller.pvalue, 0.025, rtol=1e-13)
    assert_allclose(res_tost.pvalue, 0.025, rtol=1e-13)
    x1, x2 = (x2, x1)
    res = rank_compare_2indep(x1, x2, use_t=True)
    assert_allclose(res.statistic, res2_t.statistic, rtol=1e-13)
    tt = res.test_prob_superior()
    assert_allclose(tt[0], res2_t.statistic, rtol=1e-13)
    assert_allclose(tt[1], res2_t.pvalue, rtol=1e-13)
    assert_allclose(res.pvalue, res2_t.pvalue, rtol=1e-13)
    assert_allclose(res.df, res2_t.df, rtol=1e-13)
    ci = res.conf_int(alpha=0.05)
    assert_allclose(ci, res2_t.ci, rtol=1e-11)
    res_lb = res.test_prob_superior(value=ci[0])
    assert_allclose(res_lb[1], 0.05, rtol=1e-11)
    res_ub = res.test_prob_superior(value=ci[1])
    assert_allclose(res_ub[1], 0.05, rtol=1e-11)
    res_tost = res.tost_prob_superior(ci[0], ci[1] * 1.05)
    assert_allclose(res_tost.results_larger.pvalue, 0.025, rtol=1e-10)
    assert_allclose(res_tost.pvalue, 0.025, rtol=1e-10)
    res_tost = res.tost_prob_superior(ci[0] * 0.85, ci[1])
    assert_allclose(res_tost.results_smaller.pvalue, 0.025, rtol=1e-10)
    assert_allclose(res_tost.pvalue, 0.025, rtol=1e-10)
    esd = res.effectsize_normal()
    p = prob_larger_continuous(stats.norm(loc=esd), stats.norm)
    assert_allclose(p, res.prob1, rtol=1e-13)
    pc = cohensd2problarger(esd)
    assert_allclose(pc, res.prob1, rtol=1e-13)
    ci_tr = res.confint_lintransf(1, -1)
    assert_allclose(ci_tr, 1 - np.array(res2_t.ci)[::-1], rtol=0.005)