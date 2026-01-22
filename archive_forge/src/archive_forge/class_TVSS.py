from statsmodels.compat.pandas import MONTH_END
import warnings
import numpy as np
from numpy.testing import assert_, assert_allclose
import pandas as pd
import pytest
from scipy.stats import ortho_group
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tsa.statespace import (
from statsmodels.tsa.vector_ar.tests.test_var import get_macrodata
class TVSS(mlemodel.MLEModel):
    """
    Time-varying state space model for testing

    This creates a state space model with randomly generated time-varying
    system matrices. When used in a test, that test should use
    `reset_randomstate` to ensure consistent test runs.
    """

    def __init__(self, endog, _k_states=None):
        k_states = 2
        k_posdef = 2
        if _k_states is None:
            _k_states = k_states
        super().__init__(endog, k_states=_k_states, k_posdef=k_posdef, initialization='diffuse')
        self['obs_intercept'] = np.random.normal(size=(self.k_endog, self.nobs))
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        self['transition'] = np.zeros((self.k_states, self.k_states, self.nobs))
        self['selection'] = np.zeros((self.k_states, self.ssm.k_posdef, self.nobs))
        self['design', :, :k_states, :] = np.random.normal(size=(self.k_endog, k_states, self.nobs))
        D = [np.diag(d) for d in np.random.uniform(-1.1, 1.1, size=(self.nobs, k_states))]
        Q = ortho_group.rvs(k_states, size=self.nobs)
        self['transition', :k_states, :k_states, :] = (Q @ D @ Q.transpose(0, 2, 1)).transpose(1, 2, 0)
        self['selection', :k_states, :, :] = np.random.normal(size=(k_states, self.ssm.k_posdef, self.nobs))
        H05 = np.random.normal(size=(self.k_endog, self.k_endog, self.nobs))
        Q05 = np.random.normal(size=(self.ssm.k_posdef, self.ssm.k_posdef, self.nobs))
        H = np.zeros_like(H05)
        Q = np.zeros_like(Q05)
        for t in range(self.nobs):
            H[..., t] = np.dot(H05[..., t], H05[..., t].T)
            Q[..., t] = np.dot(Q05[..., t], Q05[..., t].T)
        self['obs_cov'] = H
        self['state_cov'] = Q

    def clone(self, endog, exog=None, **kwargs):
        mod = self.__class__(endog, **kwargs)
        for key in self.ssm.shapes.keys():
            if key in ['obs', 'state_intercept']:
                continue
            n = min(self.nobs, mod.nobs)
            mod[key, ..., :n] = self.ssm[key, ..., :n]
        return mod