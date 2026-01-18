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
def test_params_effects(self):
    rng = np.random.default_rng(9121950977643805391)
    x = stats.skewnorm.rvs(-5.044559778383153, loc=1, scale=2, size=50, random_state=rng)
    guessed_params = {'c': 13.4}
    fit_params = {'scale': 13.73}
    known_params = {'loc': -13.85}
    rng = np.random.default_rng(9121950977643805391)
    res1 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2, guessed_params=guessed_params, fit_params=fit_params, known_params=known_params, random_state=rng)
    assert not np.allclose(res1.fit_result.params.c, 13.4)
    assert_equal(res1.fit_result.params.scale, 13.73)
    assert_equal(res1.fit_result.params.loc, -13.85)
    guessed_params = {'c': 2}
    rng = np.random.default_rng(9121950977643805391)
    res2 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2, guessed_params=guessed_params, fit_params=fit_params, known_params=known_params, random_state=rng)
    assert not np.allclose(res2.fit_result.params.c, res1.fit_result.params.c, rtol=1e-08)
    assert not np.allclose(res2.null_distribution, res1.null_distribution, rtol=1e-08)
    assert_equal(res2.fit_result.params.scale, 13.73)
    assert_equal(res2.fit_result.params.loc, -13.85)
    fit_params = {'c': 13.4, 'scale': 13.73}
    rng = np.random.default_rng(9121950977643805391)
    res3 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2, guessed_params=guessed_params, fit_params=fit_params, known_params=known_params, random_state=rng)
    assert_equal(res3.fit_result.params.c, 13.4)
    assert_equal(res3.fit_result.params.scale, 13.73)
    assert_equal(res3.fit_result.params.loc, -13.85)
    assert not np.allclose(res3.null_distribution, res1.null_distribution)