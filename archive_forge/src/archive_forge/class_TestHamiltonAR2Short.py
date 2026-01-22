import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression
class TestHamiltonAR2Short(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        true = {'params': np.r_[0.754673, 0.095915, -0.358811, 1.163516, np.exp(-0.262658) ** 2, 0.013486, -0.057521], 'llf': -10.14066, 'llf_fit': -4.0523073, 'llf_fit_em': -8.885836}
        super().setup_class(true, rgnp[-10:], k_regimes=2, order=2, switching_ar=False)

    def test_fit_em(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super().test_fit_em()

    def test_filter_output(self, **kwargs):
        res = self.result
        assert_allclose(res.filtered_joint_probabilities, hamilton_ar2_short_filtered_joint_probabilities)
        desired = hamilton_ar2_short_predicted_joint_probabilities
        if desired.ndim > res.predicted_joint_probabilities.ndim:
            desired = desired.sum(axis=-2)
        assert_allclose(res.predicted_joint_probabilities, desired)

    def test_smoother_output(self, **kwargs):
        res = self.result
        assert_allclose(res.filtered_joint_probabilities, hamilton_ar2_short_filtered_joint_probabilities)
        desired = hamilton_ar2_short_predicted_joint_probabilities
        if desired.ndim > res.predicted_joint_probabilities.ndim:
            desired = desired.sum(axis=-2)
        assert_allclose(res.predicted_joint_probabilities, desired)
        assert_allclose(res.smoothed_joint_probabilities[..., -1], hamilton_ar2_short_smoothed_joint_probabilities[..., -1])
        assert_allclose(res.smoothed_joint_probabilities[..., -2], hamilton_ar2_short_smoothed_joint_probabilities[..., -2])
        assert_allclose(res.smoothed_joint_probabilities[..., -3], hamilton_ar2_short_smoothed_joint_probabilities[..., -3])
        assert_allclose(res.smoothed_joint_probabilities[..., :-3], hamilton_ar2_short_smoothed_joint_probabilities[..., :-3])