import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from .._discrete_distns import nchypergeom_fisher, hypergeom
from scipy.stats._odds_ratio import odds_ratio
from .data.fisher_exact_results_from_r import data
@pytest.mark.parametrize('parameters, rresult', data)
def test_results_from_r(self, parameters, rresult):
    alternative = parameters.alternative.replace('.', '-')
    result = odds_ratio(parameters.table)
    if result.statistic < 400:
        or_rtol = 0.0005
        ci_rtol = 0.02
    else:
        or_rtol = 0.05
        ci_rtol = 0.1
    assert_allclose(result.statistic, rresult.conditional_odds_ratio, rtol=or_rtol)
    ci = result.confidence_interval(parameters.confidence_level, alternative)
    assert_allclose((ci.low, ci.high), rresult.conditional_odds_ratio_ci, rtol=ci_rtol)
    cor = result.statistic
    table = np.array(parameters.table)
    total = table.sum()
    ngood = table[0].sum()
    nsample = table[:, 0].sum()
    if cor == 0:
        nchg_mean = hypergeom.support(total, ngood, nsample)[0]
    elif cor == np.inf:
        nchg_mean = hypergeom.support(total, ngood, nsample)[1]
    else:
        nchg_mean = nchypergeom_fisher.mean(total, ngood, nsample, cor)
    assert_allclose(nchg_mean, table[0, 0], rtol=1e-13)
    alpha = 1 - parameters.confidence_level
    if alternative == 'two-sided':
        if ci.low > 0:
            sf = nchypergeom_fisher.sf(table[0, 0] - 1, total, ngood, nsample, ci.low)
            assert_allclose(sf, alpha / 2, rtol=1e-11)
        if np.isfinite(ci.high):
            cdf = nchypergeom_fisher.cdf(table[0, 0], total, ngood, nsample, ci.high)
            assert_allclose(cdf, alpha / 2, rtol=1e-11)
    elif alternative == 'less':
        if np.isfinite(ci.high):
            cdf = nchypergeom_fisher.cdf(table[0, 0], total, ngood, nsample, ci.high)
            assert_allclose(cdf, alpha, rtol=1e-11)
    elif ci.low > 0:
        sf = nchypergeom_fisher.sf(table[0, 0] - 1, total, ngood, nsample, ci.low)
        assert_allclose(sf, alpha, rtol=1e-11)