import numpy as np
import pytest
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_allclose
@pytest.mark.parametrize('missing', [None, 'mixed'])
def test_simulation_smoothing(missing):
    mod_switch = get_model(univariate=True, missing=missing)
    mod_switch.initialize_known([0], [[0]])
    sim_switch = mod_switch.simulation_smoother()
    mod_uv = get_model(univariate=True, missing=missing)
    mod_uv.initialize_known([0], [[0]])
    mod_uv.ssm.filter_univariate = True
    sim_uv = mod_uv.simulation_smoother()
    simulate_switch = mod_switch.simulate([], 10, random_state=1234)
    simulate_uv = mod_uv.simulate([], 10, random_state=1234)
    assert_allclose(simulate_switch, simulate_uv)
    sim_switch.simulate(random_state=1234)
    sim_uv.simulate(random_state=1234)
    kfilter = sim_switch._simulation_smoother.simulated_kfilter
    uf_switch = np.array(kfilter.univariate_filter, copy=True)
    assert_allclose(uf_switch[0], 1)
    assert_allclose(uf_switch[1:], 0)
    kfilter = sim_uv._simulation_smoother.simulated_kfilter.univariate_filter
    uf_uv = np.array(kfilter, copy=True)
    assert_allclose(uf_uv, 1)
    if missing == 'mixed':
        kfilter = sim_switch._simulation_smoother.secondary_simulated_kfilter.univariate_filter
        uf_switch = np.array(kfilter, copy=True)
        assert_allclose(uf_switch[0], 1)
        assert_allclose(uf_switch[1:], 0)
        kfilter = sim_uv._simulation_smoother.secondary_simulated_kfilter.univariate_filter
        uf_uv = np.array(kfilter, copy=True)
        assert_allclose(uf_uv, 1)
    attrs = ['generated_measurement_disturbance', 'generated_state_disturbance', 'generated_obs', 'generated_state', 'simulated_state', 'simulated_measurement_disturbance', 'simulated_state_disturbance']
    for attr in attrs:
        assert_allclose(getattr(sim_switch, attr), getattr(sim_uv, attr))