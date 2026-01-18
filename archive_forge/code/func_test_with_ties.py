import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.xslow()
@pytest.mark.parametrize('case', (tie_case_1, tie_case_2))
def test_with_ties(self, case):
    """
        Results above from SAS PROC NPAR1WAY, e.g.

        DATA myData;
        INPUT X Y;
        CARDS;
        1 1
        1 2
        1 3
        1 4
        2 1.5
        2 2
        2 2.5
        ods graphics on;
        proc npar1way AB data=myData;
            class X;
            EXACT;
        run;
        ods graphics off;

        Note: SAS provides Pr >= |S-Mean|, which is different from our
        definition of a two-sided p-value.

        """
    x = case['x']
    y = case['y']
    expected_statistic = case['expected_statistic']
    expected_less = case['expected_less']
    expected_2sided = case['expected_2sided']
    expected_Pr_gte_S_mean = case['expected_Pr_gte_S_mean']
    expected_avg = case['expected_avg']
    expected_std = case['expected_std']

    def statistic1d(x, y):
        return stats.ansari(x, y).statistic
    with np.testing.suppress_warnings() as sup:
        sup.filter(UserWarning, 'Ties preclude use of exact statistic')
        res = permutation_test((x, y), statistic1d, n_resamples=np.inf, alternative='less')
        res2 = permutation_test((x, y), statistic1d, n_resamples=np.inf, alternative='two-sided')
    assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
    assert_allclose(res.pvalue, expected_less, atol=1e-10)
    assert_allclose(res2.pvalue, expected_2sided, atol=1e-10)
    assert_allclose(res2.null_distribution.mean(), expected_avg, rtol=1e-06)
    assert_allclose(res2.null_distribution.std(), expected_std, rtol=1e-06)
    S = res.statistic
    mean = res.null_distribution.mean()
    n = len(res.null_distribution)
    Pr_gte_S_mean = np.sum(np.abs(res.null_distribution - mean) >= np.abs(S - mean)) / n
    assert_allclose(expected_Pr_gte_S_mean, Pr_gte_S_mean)