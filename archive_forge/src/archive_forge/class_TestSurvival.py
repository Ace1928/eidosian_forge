import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival
class TestSurvival:

    @staticmethod
    def get_random_sample(rng, n_unique):
        unique_times = rng.random(n_unique)
        repeats = rng.integers(1, 4, n_unique).astype(np.int32)
        times = rng.permuted(np.repeat(unique_times, repeats))
        censored = rng.random(size=times.size) > rng.random()
        sample = stats.CensoredData.right_censored(times, censored)
        return (sample, times, censored)

    def test_input_validation(self):
        message = '`sample` must be a one-dimensional sequence.'
        with pytest.raises(ValueError, match=message):
            stats.ecdf([[1]])
        with pytest.raises(ValueError, match=message):
            stats.ecdf(1)
        message = '`sample` must not contain nan'
        with pytest.raises(ValueError, match=message):
            stats.ecdf([np.nan])
        message = 'Currently, only uncensored and right-censored data...'
        with pytest.raises(NotImplementedError, match=message):
            stats.ecdf(stats.CensoredData.left_censored([1], censored=[True]))
        message = 'method` must be one of...'
        res = stats.ecdf([1, 2, 3])
        with pytest.raises(ValueError, match=message):
            res.cdf.confidence_interval(method='ekki-ekki')
        with pytest.raises(ValueError, match=message):
            res.sf.confidence_interval(method='shrubbery')
        message = 'confidence_level` must be a scalar between 0 and 1'
        with pytest.raises(ValueError, match=message):
            res.cdf.confidence_interval(-1)
        with pytest.raises(ValueError, match=message):
            res.sf.confidence_interval([0.5, 0.6])
        message = 'The confidence interval is undefined at some observations.'
        with pytest.warns(RuntimeWarning, match=message):
            ci = res.cdf.confidence_interval()
        message = 'Confidence interval bounds do not implement...'
        with pytest.raises(NotImplementedError, match=message):
            ci.low.confidence_interval()
        with pytest.raises(NotImplementedError, match=message):
            ci.high.confidence_interval()

    def test_edge_cases(self):
        res = stats.ecdf([])
        assert_equal(res.cdf.quantiles, [])
        assert_equal(res.cdf.probabilities, [])
        res = stats.ecdf([1])
        assert_equal(res.cdf.quantiles, [1])
        assert_equal(res.cdf.probabilities, [1])

    def test_unique(self):
        sample = [6.23, 5.58, 7.06, 6.42, 5.2]
        res = stats.ecdf(sample)
        ref_x = np.sort(np.unique(sample))
        ref_cdf = np.arange(1, 6) / 5
        ref_sf = 1 - ref_cdf
        assert_equal(res.cdf.quantiles, ref_x)
        assert_equal(res.cdf.probabilities, ref_cdf)
        assert_equal(res.sf.quantiles, ref_x)
        assert_equal(res.sf.probabilities, ref_sf)

    def test_nonunique(self):
        sample = [0, 2, 1, 2, 3, 4]
        res = stats.ecdf(sample)
        ref_x = np.sort(np.unique(sample))
        ref_cdf = np.array([1 / 6, 2 / 6, 4 / 6, 5 / 6, 1])
        ref_sf = 1 - ref_cdf
        assert_equal(res.cdf.quantiles, ref_x)
        assert_equal(res.cdf.probabilities, ref_cdf)
        assert_equal(res.sf.quantiles, ref_x)
        assert_equal(res.sf.probabilities, ref_sf)

    def test_evaluate_methods(self):
        rng = np.random.default_rng(1162729143302572461)
        sample, _, _ = self.get_random_sample(rng, 15)
        res = stats.ecdf(sample)
        x = res.cdf.quantiles
        xr = x + np.diff(x, append=x[-1] + 1) / 2
        assert_equal(res.cdf.evaluate(x), res.cdf.probabilities)
        assert_equal(res.cdf.evaluate(xr), res.cdf.probabilities)
        assert_equal(res.cdf.evaluate(x[0] - 1), 0)
        assert_equal(res.cdf.evaluate([-np.inf, np.inf]), [0, 1])
        assert_equal(res.sf.evaluate(x), res.sf.probabilities)
        assert_equal(res.sf.evaluate(xr), res.sf.probabilities)
        assert_equal(res.sf.evaluate(x[0] - 1), 1)
        assert_equal(res.sf.evaluate([-np.inf, np.inf]), [1, 0])
    t1 = [37, 43, 47, 56, 60, 62, 71, 77, 80, 81]
    d1 = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    r1 = [1, 1, 0.875, 0.75, 0.75, 0.75, 0.75, 0.5, 0.25, 0]
    t2 = [8, 12, 26, 14, 21, 27, 8, 32, 20, 40]
    d2 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    r2 = [0.9, 0.788, 0.675, 0.675, 0.54, 0.405, 0.27, 0.27, 0.27]
    t3 = [33, 28, 41, 48, 48, 25, 37, 48, 25, 43]
    d3 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    r3 = [1, 0.875, 0.75, 0.75, 0.6, 0.6, 0.6]
    t4 = [24, 3, 11, 19, 24, 13, 14, 2, 18, 17, 24, 21, 12, 1, 10, 23, 6, 5, 9, 17]
    d4 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    r4 = [0.95, 0.95, 0.897, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.76, 0.676, 0.676, 0.676, 0.676, 0.507, 0.507]
    t5 = [3, 5, 8, 10, 5, 5, 8, 12, 15, 14, 2, 11, 10, 9, 12, 5, 8, 11]
    d5 = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    r5 = [0.944, 0.889, 0.722, 0.542, 0.542, 0.542, 0.361, 0.181, 0.181, 0.181]

    @pytest.mark.parametrize('case', [(t1, d1, r1), (t2, d2, r2), (t3, d3, r3), (t4, d4, r4), (t5, d5, r5)])
    def test_right_censored_against_examples(self, case):
        times, died, ref = case
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        res = stats.ecdf(sample)
        assert_allclose(res.sf.probabilities, ref, atol=0.001)
        assert_equal(res.sf.quantiles, np.sort(np.unique(times)))
        res = _kaplan_meier_reference(times, np.logical_not(died))
        assert_equal(res[0], np.sort(np.unique(times)))
        assert_allclose(res[1], ref, atol=0.001)

    @pytest.mark.parametrize('seed', [182746786639392128, 737379171436494115, 576033618403180168, 308115465002673650])
    def test_right_censored_against_reference_implementation(self, seed):
        rng = np.random.default_rng(seed)
        n_unique = rng.integers(10, 100)
        sample, times, censored = self.get_random_sample(rng, n_unique)
        res = stats.ecdf(sample)
        ref = _kaplan_meier_reference(times, censored)
        assert_allclose(res.sf.quantiles, ref[0])
        assert_allclose(res.sf.probabilities, ref[1])
        sample = stats.CensoredData(uncensored=times)
        res = _survival._ecdf_right_censored(sample)
        ref = stats.ecdf(times)
        assert_equal(res[0], ref.sf.quantiles)
        assert_allclose(res[1], ref.cdf.probabilities, rtol=1e-14)
        assert_allclose(res[2], ref.sf.probabilities, rtol=1e-14)

    def test_right_censored_ci(self):
        times, died = (self.t4, self.d4)
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        res = stats.ecdf(sample)
        ref_allowance = [0.096, 0.096, 0.135, 0.162, 0.162, 0.162, 0.162, 0.162, 0.162, 0.162, 0.214, 0.246, 0.246, 0.246, 0.246, 0.341, 0.341]
        sf_ci = res.sf.confidence_interval()
        cdf_ci = res.cdf.confidence_interval()
        allowance = res.sf.probabilities - sf_ci.low.probabilities
        assert_allclose(allowance, ref_allowance, atol=0.001)
        assert_allclose(sf_ci.low.probabilities, np.clip(res.sf.probabilities - allowance, 0, 1))
        assert_allclose(sf_ci.high.probabilities, np.clip(res.sf.probabilities + allowance, 0, 1))
        assert_allclose(cdf_ci.low.probabilities, np.clip(res.cdf.probabilities - allowance, 0, 1))
        assert_allclose(cdf_ci.high.probabilities, np.clip(res.cdf.probabilities + allowance, 0, 1))
        ref_low = [0.694743, 0.694743, 0.647529, 0.591142, 0.591142, 0.591142, 0.591142, 0.591142, 0.591142, 0.591142, 0.464605, 0.370359, 0.370359, 0.370359, 0.370359, 0.160489, 0.160489]
        ref_high = [0.992802, 0.992802, 0.973299, 0.947073, 0.947073, 0.947073, 0.947073, 0.947073, 0.947073, 0.947073, 0.906422, 0.856521, 0.856521, 0.856521, 0.856521, 0.776724, 0.776724]
        sf_ci = res.sf.confidence_interval(method='log-log')
        assert_allclose(sf_ci.low.probabilities, ref_low, atol=1e-06)
        assert_allclose(sf_ci.high.probabilities, ref_high, atol=1e-06)

    def test_right_censored_ci_example_5(self):
        times, died = (self.t5, self.d5)
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        res = stats.ecdf(sample)
        lower = np.array([0.66639, 0.624174, 0.456179, 0.287822, 0.287822, 0.287822, 0.128489, 0.030957, 0.030957, 0.030957])
        upper = np.array([0.991983, 0.970995, 0.87378, 0.739467, 0.739467, 0.739467, 0.603133, 0.430365, 0.430365, 0.430365])
        sf_ci = res.sf.confidence_interval(method='log-log')
        cdf_ci = res.cdf.confidence_interval(method='log-log')
        assert_allclose(sf_ci.low.probabilities, lower, atol=1e-05)
        assert_allclose(sf_ci.high.probabilities, upper, atol=1e-05)
        assert_allclose(cdf_ci.low.probabilities, 1 - upper, atol=1e-05)
        assert_allclose(cdf_ci.high.probabilities, 1 - lower, atol=1e-05)
        low = [0.7436674840686117, 0.6858233228919625, 0.5059683565148012, 0.32913131413336727, 0.32913131413336727, 0.32913131413336727, 0.15986912028781664, 0.04499539918147757, 0.04499539918147757, 0.04499539918147757]
        high = [0.9890291867238429, 0.9638835422144144, 0.8560366823086629, 0.713016764397845, 0.713016764397845, 0.713016764397845, 0.5678602982997164, 0.3887616766886558, 0.3887616766886558, 0.3887616766886558]
        sf_ci = res.sf.confidence_interval(method='log-log', confidence_level=0.9)
        assert_allclose(sf_ci.low.probabilities, low)
        assert_allclose(sf_ci.high.probabilities, high)
        low = [0.8556383113628162, 0.7670478794850761, 0.5485720663578469, 0.3441515412527123, 0.3441515412527123, 0.3441515412527123, 0.1449184105424544, 0.0, 0.0, 0.0]
        high = [1.0, 1.0, 0.8958723780865975, 0.739181792080621, 0.739181792080621, 0.739181792080621, 0.5773038116797676, 0.364227025459672, 0.364227025459672, 0.364227025459672]
        sf_ci = res.sf.confidence_interval(confidence_level=0.9)
        assert_allclose(sf_ci.low.probabilities, low)
        assert_allclose(sf_ci.high.probabilities, high)

    def test_right_censored_ci_nans(self):
        times, died = (self.t1, self.d1)
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        res = stats.ecdf(sample)
        x = [37, 47, 56, 77, 80, 81]
        flo = [np.nan, 0, 0, 0.052701464070711, 0.33761112623179, np.nan]
        fup = [np.nan, 0.35417230377, 0.5500569798, 0.9472985359, 1.0, np.nan]
        i = np.searchsorted(res.cdf.quantiles, x)
        message = 'The confidence interval is undefined at some observations'
        with pytest.warns(RuntimeWarning, match=message):
            ci = res.cdf.confidence_interval()
        assert_allclose(ci.low.probabilities[i][1:], flo[1:])
        assert_allclose(ci.high.probabilities[i][1:], fup[1:])
        flo = [np.nan, 0.64582769623, 0.449943020228, 0.05270146407, 0, np.nan]
        fup = [np.nan, 1.0, 1.0, 0.947298535929289, 0.66238887376821, np.nan]
        i = np.searchsorted(res.cdf.quantiles, x)
        with pytest.warns(RuntimeWarning, match=message):
            ci = res.sf.confidence_interval()
        assert_allclose(ci.low.probabilities[i][1:], flo[1:])
        assert_allclose(ci.high.probabilities[i][1:], fup[1:])
        low = [1.0, 1.0, 0.6458276962323382, 0.44994302022779326, 0.44994302022779326, 0.44994302022779326, 0.44994302022779326, 0.05270146407071086, 0.0, np.nan]
        high = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9472985359292891, 0.6623888737682101, np.nan]
        assert_allclose(ci.low.probabilities, low)
        assert_allclose(ci.high.probabilities, high)
        with pytest.warns(RuntimeWarning, match=message):
            ci = res.sf.confidence_interval(method='log-log')
        low = [np.nan, np.nan, 0.3870000140320252, 0.3148071137055191, 0.3148071137055191, 0.3148071137055191, 0.3148071137055191, 0.08048821148507734, 0.01049958986680601, np.nan]
        high = [np.nan, np.nan, 0.981392965878966, 0.9308983170906275, 0.9308983170906275, 0.9308983170906275, 0.9308983170906275, 0.8263946341076415, 0.6558775085110887, np.nan]
        assert_allclose(ci.low.probabilities, low)
        assert_allclose(ci.high.probabilities, high)

    def test_right_censored_against_uncensored(self):
        rng = np.random.default_rng(7463952748044886637)
        sample = rng.integers(10, 100, size=1000)
        censored = np.zeros_like(sample)
        censored[np.argmax(sample)] = True
        res = stats.ecdf(sample)
        ref = stats.ecdf(stats.CensoredData.right_censored(sample, censored))
        assert_equal(res.sf.quantiles, ref.sf.quantiles)
        assert_equal(res.sf._n, ref.sf._n)
        assert_equal(res.sf._d[:-1], ref.sf._d[:-1])
        assert_allclose(res.sf._sf[:-1], ref.sf._sf[:-1], rtol=1e-14)

    def test_plot_iv(self):
        rng = np.random.default_rng(1769658657308472721)
        n_unique = rng.integers(10, 100)
        sample, _, _ = self.get_random_sample(rng, n_unique)
        res = stats.ecdf(sample)
        try:
            import matplotlib.pyplot as plt
            res.sf.plot()
        except (ModuleNotFoundError, ImportError):
            message = 'matplotlib must be installed to use method `plot`.'
            with pytest.raises(ModuleNotFoundError, match=message):
                res.sf.plot()