import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
class TestFedFundsConstShort(MarkovRegression):

    @classmethod
    def setup_class(cls):
        true = {'params': np.r_[0.9820939, 0.0503587, 3.70877, 9.556793, 2.107562 ** 2], 'llf': -29.909297, 'llf_fit': -7.855337, 'llf_fit_em': -7.8554974}
        super().setup_class(true, fedfunds[-10:], k_regimes=2)

    def test_filter_output(self, **kwargs):
        res = self.result
        assert_allclose(res.filtered_joint_probabilities, fedfunds_const_short_filtered_joint_probabilities)
        desired = fedfunds_const_short_predicted_joint_probabilities
        if desired.ndim > res.predicted_joint_probabilities.ndim:
            desired = desired.sum(axis=-2)
        assert_allclose(res.predicted_joint_probabilities, desired)

    def test_smoother_output(self, **kwargs):
        res = self.result
        assert_allclose(res.filtered_joint_probabilities, fedfunds_const_short_filtered_joint_probabilities)
        desired = fedfunds_const_short_predicted_joint_probabilities
        if desired.ndim > res.predicted_joint_probabilities.ndim:
            desired = desired.sum(axis=-2)
        assert_allclose(res.predicted_joint_probabilities, desired)
        assert_allclose(res.smoothed_joint_probabilities, fedfunds_const_short_smoothed_joint_probabilities)

    def test_hamilton_filter_order_zero(self):
        k_regimes = 3
        nobs = 4
        initial_probabilities = np.ones(k_regimes) / k_regimes
        regime_transition = np.eye(k_regimes)[:, :, np.newaxis]
        conditional_likelihoods = np.ones((k_regimes, nobs)) / 2
        conditional_likelihoods[:, 2] = [0, 1, 0]
        expected_marginals = np.empty((k_regimes, nobs))
        expected_marginals[:, :2] = [[1 / 3], [1 / 3], [1 / 3]]
        expected_marginals[:, 2:] = [[0], [1], [0]]
        cy_results = markov_switching.cy_hamilton_filter_log(initial_probabilities, regime_transition, np.log(conditional_likelihoods + 1e-20), model_order=0)
        assert_allclose(cy_results[0], expected_marginals, atol=1e-15)

    def test_hamilton_filter_order_zero_with_tvtp(self):
        k_regimes = 3
        nobs = 8
        initial_probabilities = np.ones(k_regimes) / k_regimes
        regime_transition = np.zeros((k_regimes, k_regimes, nobs))
        regime_transition[...] = np.eye(k_regimes)[:, :, np.newaxis]
        regime_transition[..., 4] = [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2], [1 / 2, 1 / 2, 1 / 2]]
        conditional_likelihoods = np.empty((k_regimes, nobs))
        conditional_likelihoods[:, 0] = [1 / 3, 1 / 3, 1 / 3]
        conditional_likelihoods[:, 1] = [1 / 3, 1 / 3, 0]
        conditional_likelihoods[:, 2] = [0, 1 / 3, 1 / 3]
        conditional_likelihoods[:, 3:5] = [[1 / 3], [1 / 3], [1 / 3]]
        conditional_likelihoods[:, 5] = [0, 1 / 3, 1 / 3]
        conditional_likelihoods[:, 6] = [0, 0, 1 / 3]
        conditional_likelihoods[:, 7] = [1 / 3, 1 / 3, 1 / 3]
        expected_marginals = np.empty((k_regimes, nobs))
        expected_marginals[:, 0] = [1 / 3, 1 / 3, 1 / 3]
        expected_marginals[:, 1] = [1 / 2, 1 / 2, 0]
        expected_marginals[:, 2:4] = [[0], [1], [0]]
        expected_marginals[:, 4:6] = [[0], [1 / 2], [1 / 2]]
        expected_marginals[:, 6:8] = [[0], [0], [1]]
        cy_results = markov_switching.cy_hamilton_filter_log(initial_probabilities, regime_transition, np.log(conditional_likelihoods + 1e-20), model_order=0)
        assert_allclose(cy_results[0], expected_marginals, atol=1e-15)

    def test_hamilton_filter_shape_checks(self):
        k_regimes = 3
        nobs = 8
        order = 3
        initial_probabilities = np.ones(k_regimes) / k_regimes
        regime_transition = np.ones((k_regimes, k_regimes, nobs)) / k_regimes
        conditional_loglikelihoods = np.ones(order * (k_regimes,) + (nobs,))
        with assert_raises(ValueError):
            markov_switching.cy_hamilton_filter_log(initial_probabilities, regime_transition, conditional_loglikelihoods, model_order=order)