import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace import dynamic_factor
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from numpy.testing import assert_equal, assert_allclose
class Trivariate:
    """
    Tests collapsing three-dimensional observation data to two-dimensional
    """

    @classmethod
    def setup_class(cls, dtype=float, alternate_timing=False, **kwargs):
        cls.results = results_kalman_filter.uc_bi
        data = pd.DataFrame(cls.results['data'], index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'), columns=['GDP', 'UNEMP'])[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = data['UNEMP'] / 100
        data['X'] = np.exp(data['GDP']) * data['UNEMP']
        k_states = 2
        cls.mlemodel = MLEModel(data, k_states=k_states, **kwargs)
        cls.model = cls.mlemodel.ssm
        if alternate_timing:
            cls.model.timing_init_filtered = True
        cls.model['selection'] = np.eye(cls.model.k_states)
        cls.model['design'] = np.array([[0.5, 0.2], [0, 0.8], [1, -0.5]])
        cls.model['transition'] = np.array([[0.4, 0.5], [1, 0]])
        cls.model['obs_cov'] = np.diag([0.2, 1.1, 0.5])
        cls.model['state_cov'] = np.diag([2.0, 1])
        cls.model.initialize_approximate_diffuse()

    def test_using_collapsed(self):
        assert not self.results_a.filter_collapsed
        assert self.results_b.filter_collapsed
        assert self.results_a.collapsed_forecasts is None
        assert self.results_b.collapsed_forecasts is not None
        assert_equal(self.results_a.forecasts.shape[0], 3)
        assert_equal(self.results_b.collapsed_forecasts.shape[0], 2)

    def test_forecasts(self):
        assert_allclose(self.results_a.forecasts[0, :], self.results_b.forecasts[0, :])

    def test_forecasts_error(self):
        assert_allclose(self.results_a.forecasts_error[0, :], self.results_b.forecasts_error[0, :])

    def test_forecasts_error_cov(self):
        assert_allclose(self.results_a.forecasts_error_cov[0, 0, :], self.results_b.forecasts_error_cov[0, 0, :])

    def test_filtered_state(self):
        assert_allclose(self.results_a.filtered_state, self.results_b.filtered_state)

    def test_filtered_state_cov(self):
        assert_allclose(self.results_a.filtered_state_cov, self.results_b.filtered_state_cov)

    def test_predicted_state(self):
        assert_allclose(self.results_a.predicted_state, self.results_b.predicted_state)

    def test_predicted_state_cov(self):
        assert_allclose(self.results_a.predicted_state_cov, self.results_b.predicted_state_cov)

    def test_loglike(self):
        assert_allclose(self.results_a.llf_obs, self.results_b.llf_obs)

    def test_smoothed_states(self):
        assert_allclose(self.results_a.smoothed_state, self.results_b.smoothed_state)

    def test_smoothed_states_cov(self):
        assert_allclose(self.results_a.smoothed_state_cov, self.results_b.smoothed_state_cov, atol=0.0001)

    def test_smoothed_states_autocov(self):
        assert_allclose(self.results_a.smoothed_state_autocov, self.results_b.smoothed_state_autocov)

    @pytest.mark.skip
    def test_smoothed_measurement_disturbance(self):
        assert_allclose(self.results_a.smoothed_measurement_disturbance, self.results_b.smoothed_measurement_disturbance)

    @pytest.mark.skip
    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(self.results_a.smoothed_measurement_disturbance_cov, self.results_b.smoothed_measurement_disturbance_cov)

    def test_smoothed_state_disturbance(self):
        assert_allclose(self.results_a.smoothed_state_disturbance, self.results_b.smoothed_state_disturbance)

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(self.results_a.smoothed_state_disturbance_cov, self.results_b.smoothed_state_disturbance_cov)

    def test_simulation_smoothed_state(self):
        assert_allclose(self.sim_a.simulated_state, self.sim_a.simulated_state)

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(self.sim_a.simulated_measurement_disturbance, self.sim_a.simulated_measurement_disturbance)

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(self.sim_a.simulated_state_disturbance, self.sim_a.simulated_state_disturbance)