import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_simulate_equivalence():
    k_groups = 4
    k_repl = 10
    nobs = np.array([10, 12, 13, 15]) * k_repl
    means = np.array([-1, 0, 0, 1]) * 0.12
    vars_ = np.array([1, 2, 3, 4])
    nobs_t = nobs.sum()
    eps = 0.0191 * 10
    opt_var = ['unequal', 'equal', 'bf']
    k_mc = 100
    np.random.seed(987126)
    res_mc = smo.simulate_power_equivalence_oneway(means, nobs, eps, vars_=vars_, k_mc=k_mc, trim_frac=0.1, options_var=opt_var, margin_type='wellek')
    frac_reject = (res_mc.pvalue <= 0.05).sum(0) / k_mc
    assert_allclose(frac_reject, [0.17, 0.18, 0.14], atol=0.001)
    es_alt_li = []
    for uv in opt_var:
        es = effectsize_oneway(means, vars_, nobs, use_var=uv)
        es_alt_li.append(es)
    margin = wellek_to_f2(eps, k_groups)
    pow_ = [power_equivalence_oneway(es_, margin, nobs_t, n_groups=k_groups, df=None, alpha=0.05, margin_type='f2') for es_ in es_alt_li]
    assert_allclose(pow_, [0.147749, 0.173358, 0.177412], atol=0.007)