import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
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