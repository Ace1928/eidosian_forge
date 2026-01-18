import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
def test_simulation_smoothing_0(self):
    sim = self.sim
    Z = self.model['design']
    nobs = self.model.nobs
    k_endog = self.model.k_endog
    k_posdef = self.model.ssm.k_posdef
    k_states = self.model.k_states
    sim.simulate(measurement_disturbance_variates=np.zeros(nobs * k_endog), state_disturbance_variates=np.zeros(nobs * k_posdef), initial_state_variates=np.zeros(k_states))
    assert_allclose(sim.generated_measurement_disturbance, 0)
    assert_allclose(sim.generated_state_disturbance, 0)
    assert_allclose(sim.generated_state, 0)
    assert_allclose(sim.generated_obs, 0)
    assert_allclose(sim.simulated_state, self.results.smoothed_state)
    if not self.model.ssm.filter_collapsed:
        assert_allclose(sim.simulated_measurement_disturbance, self.results.smoothed_measurement_disturbance)
    assert_allclose(sim.simulated_state_disturbance, self.results.smoothed_state_disturbance)
    if self.test_against_KFAS:
        path = os.path.join(current_path, 'results', 'results_simulation_smoothing0.csv')
        true = pd.read_csv(path)
        assert_allclose(sim.simulated_state, true[['state1', 'state2', 'state3']].T, atol=1e-07)
        assert_allclose(sim.simulated_measurement_disturbance, true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
        assert_allclose(sim.simulated_state_disturbance, true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
        signals = np.zeros((3, self.model.nobs))
        for t in range(self.model.nobs):
            signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
        assert_allclose(signals, true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)