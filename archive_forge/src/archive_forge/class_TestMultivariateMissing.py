import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
class TestMultivariateMissing:
    """
    Tests for most filtering and smoothing variables against output from the
    R library KFAS.

    Note that KFAS uses the univariate approach which generally will result in
    different predicted values and covariance matrices associated with the
    measurement equation (e.g. forecasts, etc.). In this case, although the
    model is multivariate, each of the series is truly independent so the
    values will be the same regardless of whether the univariate approach
    is used or not.
    """

    @classmethod
    def setup_class(cls, **kwargs):
        path = os.path.join(current_path, 'results', 'results_smoothing_R.csv')
        cls.desired = pd.read_csv(path)
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
        obs = dta[['realgdp', 'realcons', 'realinv']].diff().iloc[1:]
        obs.iloc[0:50, 0] = np.nan
        obs.iloc[19:70, 1] = np.nan
        obs.iloc[39:90, 2] = np.nan
        obs.iloc[119:130, 0] = np.nan
        obs.iloc[119:130, 2] = np.nan
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod['design'] = np.eye(3)
        mod['obs_cov'] = np.eye(3)
        mod['transition'] = np.eye(3)
        mod['selection'] = np.eye(3)
        mod['state_cov'] = np.eye(3)
        mod.initialize_approximate_diffuse(1000000.0)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        cls.results.det_scaled_smoothed_estimator_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_disturbance_cov = np.zeros((1, cls.model.nobs))
        for i in range(cls.model.nobs):
            cls.results.det_scaled_smoothed_estimator_cov[0, i] = np.linalg.det(cls.results.scaled_smoothed_estimator_cov[:, :, i])
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(cls.results.predicted_state_cov[:, :, i + 1])
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(cls.results.smoothed_state_cov[:, :, i])
            cls.results.det_smoothed_state_disturbance_cov[0, i] = np.linalg.det(cls.results.smoothed_state_disturbance_cov[:, :, i])

    def test_loglike(self):
        assert_allclose(np.sum(self.results.llf_obs), -205310.9767)

    def test_scaled_smoothed_estimator(self):
        assert_allclose(self.results.scaled_smoothed_estimator.T, self.desired[['r1', 'r2', 'r3']])

    def test_scaled_smoothed_estimator_cov(self):
        assert_allclose(self.results.det_scaled_smoothed_estimator_cov.T, self.desired[['detN']])

    def test_forecasts(self):
        assert_allclose(self.results.forecasts.T, self.desired[['m1', 'm2', 'm3']])

    def test_forecasts_error(self):
        assert_allclose(self.results.forecasts_error.T, self.desired[['v1', 'v2', 'v3']])

    def test_forecasts_error_cov(self):
        assert_allclose(self.results.forecasts_error_cov.diagonal(), self.desired[['F1', 'F2', 'F3']])

    def test_predicted_states(self):
        assert_allclose(self.results.predicted_state[:, 1:].T, self.desired[['a1', 'a2', 'a3']])

    def test_predicted_states_cov(self):
        assert_allclose(self.results.det_predicted_state_cov.T, self.desired[['detP']])

    def test_smoothed_states(self):
        assert_allclose(self.results.smoothed_state.T, self.desired[['alphahat1', 'alphahat2', 'alphahat3']])

    def test_smoothed_states_cov(self):
        assert_allclose(self.results.det_smoothed_state_cov.T, self.desired[['detV']])

    def test_smoothed_forecasts(self):
        assert_allclose(self.results.smoothed_forecasts.T, self.desired[['muhat1', 'muhat2', 'muhat3']])

    def test_smoothed_state_disturbance(self):
        assert_allclose(self.results.smoothed_state_disturbance.T, self.desired[['etahat1', 'etahat2', 'etahat3']])

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(self.results.det_smoothed_state_disturbance_cov.T, self.desired[['detVeta']])

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(self.results.smoothed_measurement_disturbance.T, self.desired[['epshat1', 'epshat2', 'epshat3']])

    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(self.results.smoothed_measurement_disturbance_cov.diagonal(), self.desired[['Veps1', 'Veps2', 'Veps3']])