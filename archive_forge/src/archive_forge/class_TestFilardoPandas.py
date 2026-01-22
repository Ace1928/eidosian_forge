import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression
class TestFilardoPandas(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        path = os.path.join(current_path, 'results', 'mar_filardo.csv')
        cls.mar_filardo = pd.read_csv(path)
        cls.mar_filardo.index = pd.date_range('1948-02-01', '1991-04-01', freq='MS')
        true = {'params': np.r_[4.35941747, -1.6493936, 1.7702123, 0.9945672, 0.517298, -0.865888, np.exp(-0.362469) ** 2, 0.189474, 0.079344, 0.110944, 0.122251], 'llf': -586.5718, 'llf_fit': -586.5718, 'llf_fit_em': -586.5718}
        endog = cls.mar_filardo['dlip'].iloc[1:]
        exog_tvtp = add_constant(cls.mar_filardo['dmdlleading'].iloc[:-1])
        super().setup_class(true, endog, k_regimes=2, order=4, switching_ar=False, exog_tvtp=exog_tvtp)

    @pytest.mark.skip
    def test_fit(self, **kwargs):
        pass

    @pytest.mark.skip
    def test_fit_em(self):
        pass

    def test_filtered_regimes(self):
        assert_allclose(self.result.filtered_marginal_probabilities[0], self.mar_filardo['filtered_0'].iloc[5:], atol=1e-05)

    def test_smoothed_regimes(self):
        assert_allclose(self.result.smoothed_marginal_probabilities[0], self.mar_filardo['smoothed_0'].iloc[5:], atol=1e-05)

    def test_expected_durations(self):
        assert_allclose(self.result.expected_durations, self.mar_filardo[['duration0', 'duration1']].iloc[5:], rtol=1e-05, atol=1e-07)