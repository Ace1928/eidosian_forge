import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
class TestPermutationTest:
    rtol = 1e-14

    def setup_method(self):
        self.rng = np.random.default_rng(7170559330470561044)

    def test_permutation_test_iv(self):

        def stat(x, y, axis):
            return stats.ttest_ind((x, y), axis).statistic
        message = 'each sample in `data` must contain two or more ...'
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1]), stat)
        message = '`data` must be a tuple containing at least two samples'
        with pytest.raises(ValueError, match=message):
            permutation_test((1,), stat)
        with pytest.raises(TypeError, match=message):
            permutation_test(1, stat)
        message = '`axis` must be an integer.'
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, axis=1.5)
        message = '`permutation_type` must be in...'
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, permutation_type='ekki')
        message = '`vectorized` must be `True`, `False`, or `None`.'
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, vectorized=1.5)
        message = '`n_resamples` must be a positive integer.'
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, n_resamples=-1000)
        message = '`n_resamples` must be a positive integer.'
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, n_resamples=1000.5)
        message = '`batch` must be a positive integer or None.'
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, batch=-1000)
        message = '`batch` must be a positive integer or None.'
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, batch=1000.5)
        message = '`alternative` must be in...'
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, alternative='ekki')
        message = "'herring' cannot be used to seed a"
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, random_state='herring')

    @pytest.mark.parametrize('random_state', [np.random.RandomState, np.random.default_rng])
    @pytest.mark.parametrize('permutation_type', ['pairings', 'samples', 'independent'])
    def test_batch(self, permutation_type, random_state):
        x = self.rng.random(10)
        y = self.rng.random(10)

        def statistic(x, y, axis):
            batch_size = 1 if x.ndim == 1 else len(x)
            statistic.batch_size = max(batch_size, statistic.batch_size)
            statistic.counter += 1
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        statistic.counter = 0
        statistic.batch_size = 0
        kwds = {'n_resamples': 1000, 'permutation_type': permutation_type, 'vectorized': True}
        res1 = stats.permutation_test((x, y), statistic, batch=1, random_state=random_state(0), **kwds)
        assert_equal(statistic.counter, 1001)
        assert_equal(statistic.batch_size, 1)
        statistic.counter = 0
        res2 = stats.permutation_test((x, y), statistic, batch=50, random_state=random_state(0), **kwds)
        assert_equal(statistic.counter, 21)
        assert_equal(statistic.batch_size, 50)
        statistic.counter = 0
        res3 = stats.permutation_test((x, y), statistic, batch=1000, random_state=random_state(0), **kwds)
        assert_equal(statistic.counter, 2)
        assert_equal(statistic.batch_size, 1000)
        assert_equal(res1.pvalue, res3.pvalue)
        assert_equal(res2.pvalue, res3.pvalue)

    @pytest.mark.parametrize('random_state', [np.random.RandomState, np.random.default_rng])
    @pytest.mark.parametrize('permutation_type, exact_size', [('pairings', special.factorial(3) ** 2), ('samples', 2 ** 3), ('independent', special.binom(6, 3))])
    def test_permutations(self, permutation_type, exact_size, random_state):
        x = self.rng.random(3)
        y = self.rng.random(3)

        def statistic(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        kwds = {'permutation_type': permutation_type, 'vectorized': True}
        res = stats.permutation_test((x, y), statistic, n_resamples=3, random_state=random_state(0), **kwds)
        assert_equal(res.null_distribution.size, 3)
        res = stats.permutation_test((x, y), statistic, **kwds)
        assert_equal(res.null_distribution.size, exact_size)

    def test_randomized_test_against_exact_both(self):
        alternative, rng = ('less', 0)
        nx, ny, permutations = (8, 9, 24000)
        assert special.binom(nx + ny, nx) > permutations
        x = stats.norm.rvs(size=nx)
        y = stats.norm.rvs(size=ny)
        data = (x, y)

        def statistic(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        kwds = {'vectorized': True, 'permutation_type': 'independent', 'batch': 100, 'alternative': alternative, 'random_state': rng}
        res = permutation_test(data, statistic, n_resamples=permutations, **kwds)
        res2 = permutation_test(data, statistic, n_resamples=np.inf, **kwds)
        assert res.statistic == res2.statistic
        assert_allclose(res.pvalue, res2.pvalue, atol=0.01)

    @pytest.mark.slow()
    def test_randomized_test_against_exact_samples(self):
        alternative, rng = ('greater', None)
        nx, ny, permutations = (15, 15, 32000)
        assert 2 ** nx > permutations
        x = stats.norm.rvs(size=nx)
        y = stats.norm.rvs(size=ny)
        data = (x, y)

        def statistic(x, y, axis):
            return np.mean(x - y, axis=axis)
        kwds = {'vectorized': True, 'permutation_type': 'samples', 'batch': 100, 'alternative': alternative, 'random_state': rng}
        res = permutation_test(data, statistic, n_resamples=permutations, **kwds)
        res2 = permutation_test(data, statistic, n_resamples=np.inf, **kwds)
        assert res.statistic == res2.statistic
        assert_allclose(res.pvalue, res2.pvalue, atol=0.01)

    def test_randomized_test_against_exact_pairings(self):
        alternative, rng = ('two-sided', self.rng)
        nx, ny, permutations = (8, 8, 40000)
        assert special.factorial(nx) > permutations
        x = stats.norm.rvs(size=nx)
        y = stats.norm.rvs(size=ny)
        data = [x]

        def statistic1d(x):
            return stats.pearsonr(x, y)[0]
        statistic = _resampling._vectorize_statistic(statistic1d)
        kwds = {'vectorized': True, 'permutation_type': 'samples', 'batch': 100, 'alternative': alternative, 'random_state': rng}
        res = permutation_test(data, statistic, n_resamples=permutations, **kwds)
        res2 = permutation_test(data, statistic, n_resamples=np.inf, **kwds)
        assert res.statistic == res2.statistic
        assert_allclose(res.pvalue, res2.pvalue, atol=0.01)

    @pytest.mark.parametrize('alternative', ('less', 'greater'))
    @pytest.mark.parametrize('permutations', (30, 1000000000.0))
    @pytest.mark.parametrize('axis', (0, 1, 2))
    def test_against_permutation_ttest(self, alternative, permutations, axis):
        x = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        y = np.moveaxis(np.arange(4)[:, None, None], 0, axis)
        rng1 = np.random.default_rng(4337234444626115331)
        res1 = stats.ttest_ind(x, y, permutations=permutations, axis=axis, random_state=rng1, alternative=alternative)

        def statistic(x, y, axis):
            return stats.ttest_ind(x, y, axis=axis).statistic
        rng2 = np.random.default_rng(4337234444626115331)
        res2 = permutation_test((x, y), statistic, vectorized=True, n_resamples=permutations, alternative=alternative, axis=axis, random_state=rng2)
        assert_allclose(res1.statistic, res2.statistic, rtol=self.rtol)
        assert_allclose(res1.pvalue, res2.pvalue, rtol=self.rtol)

    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    def test_against_ks_2samp(self, alternative):
        x = self.rng.normal(size=4, scale=1)
        y = self.rng.normal(size=5, loc=3, scale=3)
        expected = stats.ks_2samp(x, y, alternative=alternative, mode='exact')

        def statistic1d(x, y):
            return stats.ks_2samp(x, y, mode='asymp', alternative=alternative).statistic
        res = permutation_test((x, y), statistic1d, n_resamples=np.inf, alternative='greater', random_state=self.rng)
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    def test_against_ansari(self, alternative):
        x = self.rng.normal(size=4, scale=1)
        y = self.rng.normal(size=5, scale=3)
        alternative_correspondence = {'less': 'greater', 'greater': 'less', 'two-sided': 'two-sided'}
        alternative_scipy = alternative_correspondence[alternative]
        expected = stats.ansari(x, y, alternative=alternative_scipy)

        def statistic1d(x, y):
            return stats.ansari(x, y).statistic
        res = permutation_test((x, y), statistic1d, n_resamples=np.inf, alternative=alternative, random_state=self.rng)
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    def test_against_mannwhitneyu(self, alternative):
        x = stats.uniform.rvs(size=(3, 5, 2), loc=0, random_state=self.rng)
        y = stats.uniform.rvs(size=(3, 5, 2), loc=0.05, random_state=self.rng)
        expected = stats.mannwhitneyu(x, y, axis=1, alternative=alternative)

        def statistic(x, y, axis):
            return stats.mannwhitneyu(x, y, axis=axis).statistic
        res = permutation_test((x, y), statistic, vectorized=True, n_resamples=np.inf, alternative=alternative, axis=1, random_state=self.rng)
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    def test_against_cvm(self):
        x = stats.norm.rvs(size=4, scale=1, random_state=self.rng)
        y = stats.norm.rvs(size=5, loc=3, scale=3, random_state=self.rng)
        expected = stats.cramervonmises_2samp(x, y, method='exact')

        def statistic1d(x, y):
            return stats.cramervonmises_2samp(x, y, method='asymptotic').statistic
        res = permutation_test((x, y), statistic1d, n_resamples=np.inf, alternative='greater', random_state=self.rng)
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    @pytest.mark.xslow()
    @pytest.mark.parametrize('axis', (-1, 2))
    def test_vectorized_nsamp_ptype_both(self, axis):
        rng = np.random.default_rng(6709265303529651545)
        x = rng.random(size=3)
        y = rng.random(size=(1, 3, 2))
        z = rng.random(size=(2, 1, 4))
        data = (x, y, z)

        def statistic1d(*data):
            return stats.kruskal(*data).statistic

        def pvalue1d(*data):
            return stats.kruskal(*data).pvalue
        statistic = _resampling._vectorize_statistic(statistic1d)
        pvalue = _resampling._vectorize_statistic(pvalue1d)
        x2 = np.broadcast_to(x, (2, 3, 3))
        y2 = np.broadcast_to(y, (2, 3, 2))
        z2 = np.broadcast_to(z, (2, 3, 4))
        expected_statistic = statistic(x2, y2, z2, axis=axis)
        expected_pvalue = pvalue(x2, y2, z2, axis=axis)
        kwds = {'vectorized': False, 'axis': axis, 'alternative': 'greater', 'permutation_type': 'independent', 'random_state': self.rng}
        res = permutation_test(data, statistic1d, n_resamples=np.inf, **kwds)
        res2 = permutation_test(data, statistic1d, n_resamples=1000, **kwds)
        assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
        assert_allclose(res.statistic, res2.statistic, rtol=self.rtol)
        assert_allclose(res.pvalue, expected_pvalue, atol=0.06)
        assert_allclose(res.pvalue, res2.pvalue, atol=0.03)

    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    def test_against_wilcoxon(self, alternative):
        x = stats.uniform.rvs(size=(3, 6, 2), loc=0, random_state=self.rng)
        y = stats.uniform.rvs(size=(3, 6, 2), loc=0.05, random_state=self.rng)

        def statistic_1samp_1d(z):
            return stats.wilcoxon(z, alternative='less').statistic

        def statistic_2samp_1d(x, y):
            return stats.wilcoxon(x, y, alternative='less').statistic

        def test_1d(x, y):
            return stats.wilcoxon(x, y, alternative=alternative)
        test = _resampling._vectorize_statistic(test_1d)
        expected = test(x, y, axis=1)
        expected_stat = expected[0]
        expected_p = expected[1]
        kwds = {'vectorized': False, 'axis': 1, 'alternative': alternative, 'permutation_type': 'samples', 'random_state': self.rng, 'n_resamples': np.inf}
        res1 = permutation_test((x - y,), statistic_1samp_1d, **kwds)
        res2 = permutation_test((x, y), statistic_2samp_1d, **kwds)
        assert_allclose(res1.statistic, res2.statistic, rtol=self.rtol)
        if alternative != 'two-sided':
            assert_allclose(res2.statistic, expected_stat, rtol=self.rtol)
        assert_allclose(res2.pvalue, expected_p, rtol=self.rtol)
        assert_allclose(res1.pvalue, res2.pvalue, rtol=self.rtol)

    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    def test_against_binomtest(self, alternative):
        x = self.rng.integers(0, 2, size=10)
        x[x == 0] = -1

        def statistic(x, axis=0):
            return np.sum(x > 0, axis=axis)
        k, n, p = (statistic(x), 10, 0.5)
        expected = stats.binomtest(k, n, p, alternative=alternative)
        res = stats.permutation_test((x,), statistic, vectorized=True, permutation_type='samples', n_resamples=np.inf, random_state=self.rng, alternative=alternative)
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    def test_against_kendalltau(self):
        x = self.rng.normal(size=6)
        y = x + self.rng.normal(size=6)
        expected = stats.kendalltau(x, y, method='exact')

        def statistic1d(x):
            return stats.kendalltau(x, y, method='asymptotic').statistic
        res = permutation_test((x,), statistic1d, permutation_type='pairings', n_resamples=np.inf, random_state=self.rng)
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    def test_against_fisher_exact(self, alternative):

        def statistic(x):
            return np.sum((x == 1) & (y == 1))
        rng = np.random.default_rng(6235696159000529929)
        x = (rng.random(7) > 0.6).astype(float)
        y = (rng.random(7) + 0.25 * x > 0.6).astype(float)
        tab = stats.contingency.crosstab(x, y)[1]
        res = permutation_test((x,), statistic, permutation_type='pairings', n_resamples=np.inf, alternative=alternative, random_state=rng)
        res2 = stats.fisher_exact(tab, alternative=alternative)
        assert_allclose(res.pvalue, res2[1])

    @pytest.mark.xslow()
    @pytest.mark.parametrize('axis', (-2, 1))
    def test_vectorized_nsamp_ptype_samples(self, axis):
        x = self.rng.random(size=(2, 4, 3))
        y = self.rng.random(size=(1, 4, 3))
        z = self.rng.random(size=(2, 4, 1))
        x = stats.rankdata(x, axis=axis)
        y = stats.rankdata(y, axis=axis)
        z = stats.rankdata(z, axis=axis)
        y = y[0]
        data = (x, y, z)

        def statistic1d(*data):
            return stats.page_trend_test(data, ranked=True, method='asymptotic').statistic

        def pvalue1d(*data):
            return stats.page_trend_test(data, ranked=True, method='exact').pvalue
        statistic = _resampling._vectorize_statistic(statistic1d)
        pvalue = _resampling._vectorize_statistic(pvalue1d)
        expected_statistic = statistic(*np.broadcast_arrays(*data), axis=axis)
        expected_pvalue = pvalue(*np.broadcast_arrays(*data), axis=axis)
        kwds = {'vectorized': False, 'axis': axis, 'alternative': 'greater', 'permutation_type': 'pairings', 'random_state': 0}
        res = permutation_test(data, statistic1d, n_resamples=np.inf, **kwds)
        res2 = permutation_test(data, statistic1d, n_resamples=5000, **kwds)
        assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
        assert_allclose(res.statistic, res2.statistic, rtol=self.rtol)
        assert_allclose(res.pvalue, expected_pvalue, rtol=self.rtol)
        assert_allclose(res.pvalue, res2.pvalue, atol=0.03)
    tie_case_1 = {'x': [1, 2, 3, 4], 'y': [1.5, 2, 2.5], 'expected_less': 0.2, 'expected_2sided': 0.4, 'expected_Pr_gte_S_mean': 0.3428571429, 'expected_statistic': 7.5, 'expected_avg': 9.142857, 'expected_std': 1.40698}
    tie_case_2 = {'x': [111, 107, 100, 99, 102, 106, 109, 108], 'y': [107, 108, 106, 98, 105, 103, 110, 105, 104], 'expected_less': 0.1555738379, 'expected_2sided': 0.3111476758, 'expected_Pr_gte_S_mean': 0.2969971205, 'expected_statistic': 32.5, 'expected_avg': 38.117647, 'expected_std': 5.172124}

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

    @pytest.mark.parametrize('alternative, expected_pvalue', (('less', 0.9708333333333), ('greater', 0.05138888888889), ('two-sided', 0.1027777777778)))
    def test_against_spearmanr_in_R(self, alternative, expected_pvalue):
        """
        Results above from R cor.test, e.g.

        options(digits=16)
        x <- c(1.76405235, 0.40015721, 0.97873798,
               2.2408932, 1.86755799, -0.97727788)
        y <- c(2.71414076, 0.2488, 0.87551913,
               2.6514917, 2.01160156, 0.47699563)
        cor.test(x, y, method = "spearm", alternative = "t")
        """
        x = [1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799, -0.97727788]
        y = [2.71414076, 0.2488, 0.87551913, 2.6514917, 2.01160156, 0.47699563]
        expected_statistic = 0.7714285714285715

        def statistic1d(x):
            return stats.spearmanr(x, y).statistic
        res = permutation_test((x,), statistic1d, permutation_type='pairings', n_resamples=np.inf, alternative=alternative)
        assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
        assert_allclose(res.pvalue, expected_pvalue, atol=1e-13)

    @pytest.mark.parametrize('batch', (-1, 0))
    def test_batch_generator_iv(self, batch):
        with pytest.raises(ValueError, match='`batch` must be positive.'):
            list(_resampling._batch_generator([1, 2, 3], batch))
    batch_generator_cases = [(range(0), 3, []), (range(6), 3, [[0, 1, 2], [3, 4, 5]]), (range(8), 3, [[0, 1, 2], [3, 4, 5], [6, 7]])]

    @pytest.mark.parametrize('iterable, batch, expected', batch_generator_cases)
    def test_batch_generator(self, iterable, batch, expected):
        got = list(_resampling._batch_generator(iterable, batch))
        assert got == expected

    def test_finite_precision_statistic(self):
        x = [1, 2, 4, 3]
        y = [2, 4, 6, 8]

        def statistic(x, y):
            return stats.pearsonr(x, y)[0]
        res = stats.permutation_test((x, y), statistic, vectorized=False, permutation_type='pairings')
        r, pvalue, null = (res.statistic, res.pvalue, res.null_distribution)
        correct_p = 2 * np.sum(null >= r - 1e-14) / len(null)
        assert pvalue == correct_p == 1 / 3