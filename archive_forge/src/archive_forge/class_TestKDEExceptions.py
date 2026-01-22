import os
import numpy.testing as npt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import statsmodels.sandbox.nonparametric.kernels as kernels
import statsmodels.nonparametric.bandwidths as bandwidths
class TestKDEExceptions:

    @classmethod
    def setup_class(cls):
        cls.kde = KDE(Xi)
        cls.weights_200 = np.linspace(1, 100, 200)
        cls.weights_100 = np.linspace(1, 100, 100)

    def test_check_is_fit_exception(self):
        with pytest.raises(ValueError):
            self.kde.evaluate(0)

    def test_non_weighted_fft_exception(self):
        with pytest.raises(NotImplementedError):
            self.kde.fit(kernel='gau', gridsize=50, weights=self.weights_200, fft=True, bw='silverman')

    def test_wrong_weight_length_exception(self):
        with pytest.raises(ValueError):
            self.kde.fit(kernel='gau', gridsize=50, weights=self.weights_100, fft=False, bw='silverman')

    def test_non_gaussian_fft_exception(self):
        with pytest.raises(NotImplementedError):
            self.kde.fit(kernel='epa', gridsize=50, fft=True, bw='silverman')