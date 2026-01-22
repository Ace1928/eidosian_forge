import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
class TestFedFundsConst(MarkovRegression):

    @classmethod
    def setup_class(cls):
        path = os.path.join(current_path, 'results', 'results_predict_fedfunds.csv')
        results = pd.read_csv(path)
        true = {'params': np.r_[0.9820939, 0.0503587, 3.70877, 9.556793, 2.107562 ** 2], 'llf': -508.63592, 'llf_fit': -508.63592, 'llf_fit_em': -508.65852, 'bse_oim': np.r_[0.0104002, 0.0268434, 0.1767083, 0.2999889, np.nan], 'smoothed0': results['const_sm1'], 'smoothed1': results['const_sm2'], 'predict0': results['const_yhat1'], 'predict1': results['const_yhat2'], 'predict_predicted': results['const_pyhat'], 'predict_filtered': results['const_fyhat'], 'predict_smoothed': results['const_syhat']}
        super().setup_class(true, fedfunds, k_regimes=2)

    def test_filter_output(self, **kwargs):
        res = self.result
        assert_allclose(res.filtered_joint_probabilities, fedfunds_const_filtered_joint_probabilities)

    def test_smoothed_marginal_probabilities(self):
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 0], self.true['smoothed0'], atol=1e-06)
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 1], self.true['smoothed1'], atol=1e-06)

    def test_predict(self):
        for name in ['predicted', 'filtered', 'smoothed', None]:
            actual = self.model.predict(self.true['params'], probabilities=name, conditional=True)
            assert_allclose(actual[0], self.true['predict0'], atol=1e-06)
            assert_allclose(actual[1], self.true['predict1'], atol=1e-06)
        actual = self.model.predict(self.true['params'], probabilities='predicted')
        assert_allclose(actual, self.true['predict_predicted'], atol=1e-05)
        actual = self.model.predict(self.true['params'], probabilities='filtered')
        assert_allclose(self.model.predict(self.true['params'], probabilities='filtered'), self.true['predict_filtered'], atol=1e-05)
        actual = self.model.predict(self.true['params'], probabilities='smoothed')
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-06)
        actual = self.model.predict(self.true['params'], probabilities=None)
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-06)

    def test_bse(self):
        bse = self.result.cov_params_approx.diagonal() ** 0.5
        assert_allclose(bse[:-1], self.true['bse_oim'][:-1], atol=1e-07)