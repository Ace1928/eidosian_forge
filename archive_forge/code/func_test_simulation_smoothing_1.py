import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
def test_simulation_smoothing_1(self):
    sim = self.sim
    Z = self.model['design']
    nobs = self.model.nobs
    k_endog = self.model.k_endog
    k_posdef = self.model.ssm.k_posdef
    k_states = self.model.k_states
    measurement_disturbance_variates = np.reshape(np.arange(nobs * k_endog) / 10.0, (nobs, k_endog))
    state_disturbance_variates = np.zeros(nobs * k_posdef)
    generated_measurement_disturbance = np.zeros(measurement_disturbance_variates.shape)
    chol = np.linalg.cholesky(self.model['obs_cov'])
    for t in range(self.model.nobs):
        generated_measurement_disturbance[t] = np.dot(chol, measurement_disturbance_variates[t])
    y = generated_measurement_disturbance.copy()
    y[np.isnan(self.model.endog)] = np.nan
    generated_model = mlemodel.MLEModel(y, k_states=k_states, k_posdef=k_posdef)
    for name in ['design', 'obs_cov', 'transition', 'selection', 'state_cov']:
        generated_model[name] = self.model[name]
    generated_model.initialize_approximate_diffuse(1000000.0)
    generated_model.ssm.filter_univariate = True
    generated_res = generated_model.ssm.smooth()
    simulated_state = 0 - generated_res.smoothed_state + self.results.smoothed_state
    if not self.model.ssm.filter_collapsed:
        simulated_measurement_disturbance = generated_measurement_disturbance.T - generated_res.smoothed_measurement_disturbance + self.results.smoothed_measurement_disturbance
    simulated_state_disturbance = 0 - generated_res.smoothed_state_disturbance + self.results.smoothed_state_disturbance
    sim.simulate(measurement_disturbance_variates=measurement_disturbance_variates, state_disturbance_variates=state_disturbance_variates, initial_state_variates=np.zeros(k_states))
    assert_allclose(sim.generated_measurement_disturbance, generated_measurement_disturbance)
    assert_allclose(sim.generated_state_disturbance, 0)
    assert_allclose(sim.generated_state, 0)
    assert_allclose(sim.generated_obs, generated_measurement_disturbance.T)
    assert_allclose(sim.simulated_state, simulated_state)
    if not self.model.ssm.filter_collapsed:
        assert_allclose(sim.simulated_measurement_disturbance, simulated_measurement_disturbance)
    assert_allclose(sim.simulated_state_disturbance, simulated_state_disturbance)
    if self.test_against_KFAS:
        path = os.path.join(current_path, 'results', 'results_simulation_smoothing1.csv')
        true = pd.read_csv(path)
        assert_allclose(sim.simulated_state, true[['state1', 'state2', 'state3']].T, atol=1e-07)
        assert_allclose(sim.simulated_measurement_disturbance, true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
        assert_allclose(sim.simulated_state_disturbance, true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
        signals = np.zeros((3, self.model.nobs))
        for t in range(self.model.nobs):
            signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
        assert_allclose(signals, true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)