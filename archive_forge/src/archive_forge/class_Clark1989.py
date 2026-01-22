import os
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother
from statsmodels.tsa.statespace import tools, sarimax
from .results import results_kalman_filter
from numpy.testing import (
class Clark1989:
    """
    Clark's (1989) bivariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Tests two-dimensional observation data.

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """

    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        cls.true = results_kalman_filter.uc_bi
        cls.true_states = pd.DataFrame(cls.true['states'])
        data = pd.DataFrame(cls.true['data'], index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'), columns=['GDP', 'UNEMP'])[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = data['UNEMP'] / 100
        k_states = 6
        cls.model = KalmanFilter(k_endog=2, k_states=k_states, **kwargs)
        cls.model.bind(np.ascontiguousarray(data.values))
        cls.model.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        cls.model.transition[[0, 0, 1, 1, 2, 3, 4, 5], [0, 4, 1, 2, 1, 2, 4, 5], [0, 0, 0, 0, 0, 0, 0, 0]] = [1, 1, 0, 0, 1, 1, 1, 1]
        cls.model.selection = np.eye(cls.model.k_states)
        sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec, phi_1, phi_2, alpha_1, alpha_2, alpha_3 = np.array(cls.true['parameters'])
        cls.model.design[[1, 1, 1], [1, 2, 3], [0, 0, 0]] = [alpha_1, alpha_2, alpha_3]
        cls.model.transition[[1, 1], [1, 2], [0, 0]] = [phi_1, phi_2]
        cls.model.obs_cov[1, 1, 0] = sigma_ec ** 2
        cls.model.state_cov[np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)] = [sigma_v ** 2, sigma_e ** 2, 0, 0, sigma_w ** 2, sigma_vl ** 2]
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states) * 100
        initial_state_cov = np.dot(np.dot(cls.model.transition[:, :, 0], initial_state_cov), cls.model.transition[:, :, 0].T)
        cls.model.initialize_known(initial_state, initial_state_cov)

    @classmethod
    def run_filter(cls):
        return cls.model.filter()

    def test_loglike(self):
        assert_almost_equal(self.results.llf_obs[0:].sum(), self.true['loglike'], 2)

    def test_filtered_state(self):
        assert_almost_equal(self.results.filtered_state[0][self.true['start']:], self.true_states.iloc[:, 0], 4)
        assert_almost_equal(self.results.filtered_state[1][self.true['start']:], self.true_states.iloc[:, 1], 4)
        assert_almost_equal(self.results.filtered_state[4][self.true['start']:], self.true_states.iloc[:, 2], 4)
        assert_almost_equal(self.results.filtered_state[5][self.true['start']:], self.true_states.iloc[:, 3], 4)