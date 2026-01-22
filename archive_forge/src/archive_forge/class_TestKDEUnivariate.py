import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
class TestKDEUnivariate(KDETestBase):

    def test_pdf_non_fft(self):
        kde = nparam.KDEUnivariate(self.noise)
        kde.fit(fft=False, bw='scott')
        grid = kde.support
        testx = [grid[10 * i] for i in range(6)]
        kde_expected = [0.00016808277984236013, 0.030759614592368954, 0.14123404934759243, 0.2880714740816241, 0.25594519303876273, 0.05659397391565105]
        kde_vals0 = kde.density[10 * np.arange(6)]
        kde_vals = kde.evaluate(testx)
        npt.assert_allclose(kde_vals, kde_expected, atol=1e-06)
        npt.assert_allclose(kde_vals0, kde_expected, atol=1e-06)

    def test_weighted_pdf_non_fft(self):
        kde = nparam.KDEUnivariate(self.noise)
        kde.fit(weights=self.weights, fft=False, bw='scott')
        grid = kde.support
        testx = [grid[10 * i] for i in range(6)]
        kde_expected = [9.199885803395076e-05, 0.018761981151370496, 0.14425925509365087, 0.30307631742267443, 0.2405445849994125, 0.06433170684797665]
        kde_vals0 = kde.density[10 * np.arange(6)]
        kde_vals = kde.evaluate(testx)
        npt.assert_allclose(kde_vals, kde_expected, atol=1e-06)
        npt.assert_allclose(kde_vals0, kde_expected, atol=1e-06)

    def test_all_samples_same_location_bw(self):
        x = np.ones(100)
        kde = nparam.KDEUnivariate(x)
        with pytest.raises(RuntimeError, match='Selected KDE bandwidth is 0'):
            kde.fit()

    def test_int(self, reset_randomstate):
        x = np.random.randint(0, 100, size=1000)
        kde = nparam.KDEUnivariate(x)
        kde.fit()
        kde_double = nparam.KDEUnivariate(x.astype('double'))
        kde_double.fit()
        assert_allclose(kde.bw, kde_double.bw)