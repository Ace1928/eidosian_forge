from scipy import stats
from numpy.testing import assert_allclose
from statsmodels.stats.effect_size import (
def test_noncent_t():
    t_stat, df = (1.5, 98)
    ci_nc = [-0.474934, 3.467371]
    res = _noncentrality_t(t_stat, df, alpha=0.05)
    assert_allclose(res.confint, ci_nc, rtol=0.005)
    mean = stats.nct.mean(df, res.nc)
    assert_allclose(t_stat, mean, rtol=1e-08)
    assert_allclose(stats.nct.cdf(t_stat, df, res.confint), [0.975, 0.025], rtol=1e-06)