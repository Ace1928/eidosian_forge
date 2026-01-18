import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, dynamic_factor,
def test_posterior_cov(self):
    inv_chol = np.array(self._sim_cfa.posterior_cov_inv_chol, copy=True)
    actual = cho_solve_banded((inv_chol, True), np.eye(inv_chol.shape[1]))
    for t in range(self.mod.nobs):
        tm = t * self.mod.k_states
        t1m = tm + self.mod.k_states
        assert_allclose(actual[tm:t1m, tm:t1m], self.res.smoothed_state_cov[..., t], atol=self.cov_atol)
    actual = self.sim_cfa.posterior_cov
    for t in range(self.mod.nobs):
        tm = t * self.mod.k_states
        t1m = tm + self.mod.k_states
        assert_allclose(actual[tm:t1m, tm:t1m], self.res.smoothed_state_cov[..., t], atol=self.cov_atol)