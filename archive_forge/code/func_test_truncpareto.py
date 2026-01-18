import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
def test_truncpareto(self):
    N = 1000
    rng = np.random.default_rng(self.seed)
    dist = stats.truncpareto
    shapes = (1.8, 5.3, 2.3, 4.1)
    data = dist.rvs(*shapes, size=N, random_state=rng)
    shape_bounds = [(0.1, 10)] * 4
    res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)
    assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)