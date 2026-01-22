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
class TestVARAutocovariances:

    @classmethod
    def setup_class(cls, which='mixed', *args, **kwargs):
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
        obs = np.log(dta[['realgdp', 'realcons', 'realinv']]).diff().iloc[1:]
        if which == 'all':
            obs.iloc[:50, :] = np.nan
            obs.iloc[119:130, :] = np.nan
        elif which == 'partial':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[119:130, 0] = np.nan
        elif which == 'mixed':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod['design'] = np.eye(3)
        mod['obs_cov'] = np.array([[609.0746647855, 0.0, 0.0], [0.0, 1.8774916622, 0.0], [0.0, 0.0, 124.6768281675]])
        mod['transition'] = np.array([[-0.8110473405, 1.8005304445, 1.0215975772], [-1.9846632699, 2.4091302213, 1.9264449765], [0.9181658823, -0.2442384581, -0.6393462272]])
        mod['selection'] = np.eye(3)
        mod['state_cov'] = np.array([[1552.9758843938, 612.7185121905, 877.6157204992], [612.7185121905, 467.8739411204, 70.608037339], [877.6157204992, 70.608037339, 900.5440385836]])
        mod.initialize_approximate_diffuse(1000000.0)
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        kwargs.pop('filter_collapsed', None)
        mod = mlemodel.MLEModel(obs, k_states=6, k_posdef=3, **kwargs)
        mod['design', :3, :3] = np.eye(3)
        mod['obs_cov'] = np.array([[609.0746647855, 0.0, 0.0], [0.0, 1.8774916622, 0.0], [0.0, 0.0, 124.6768281675]])
        mod['transition', :3, :3] = np.array([[-0.8110473405, 1.8005304445, 1.0215975772], [-1.9846632699, 2.4091302213, 1.9264449765], [0.9181658823, -0.2442384581, -0.6393462272]])
        mod['transition', 3:, :3] = np.eye(3)
        mod['selection', :3, :3] = np.eye(3)
        mod['state_cov'] = np.array([[1552.9758843938, 612.7185121905, 877.6157204992], [612.7185121905, 467.8739411204, 70.608037339], [877.6157204992, 70.608037339, 900.5440385836]])
        mod.initialize_approximate_diffuse(1000000.0)
        cls.augmented_model = mod
        cls.augmented_results = mod.smooth([], return_ssm=True)

    def test_smoothed_state_autocov(self):
        assert_allclose(self.results.smoothed_state_autocov[:, :, 0:5], self.augmented_results.smoothed_state_cov[:3, 3:, 1:6], atol=0.0001)
        assert_allclose(self.results.smoothed_state_autocov[:, :, 5:-1], self.augmented_results.smoothed_state_cov[:3, 3:, 6:], atol=1e-07)