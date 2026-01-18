from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_structural():
    np.random.seed(38947)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=nobs)
    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1
    mod1 = structural.UnobservedComponents([0], autoregressive=1)
    mod2 = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod1.simulate([1, 0.5], nobs, state_shocks=eps, initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)
    mod1 = structural.UnobservedComponents(np.zeros(nobs), exog=exog, autoregressive=1)
    mod2 = sarimax.SARIMAX(np.zeros(nobs), exog=exog, order=(1, 0, 0))
    actual = mod1.simulate([1, 0.5, 0.2], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states))
    desired = mod2.simulate([0.2, 0.5, 1], nobs, state_shocks=eps, initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'irregular')
    actual = mod.simulate([1.0], nobs, measurement_shocks=eps, initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, eps)
    warning = SpecificationWarning
    match = 'irregular component added'
    with pytest.warns(warning, match=match):
        mod = structural.UnobservedComponents([0], 'fixed intercept')
    actual = mod.simulate([1.0], nobs, measurement_shocks=eps, initial_state=[10])
    assert_allclose(actual, 10 + eps)
    mod = structural.UnobservedComponents([0], 'deterministic constant')
    actual = mod.simulate([1.0], nobs, measurement_shocks=eps, initial_state=[10])
    assert_allclose(actual, 10 + eps)
    mod = structural.UnobservedComponents([0], 'local level')
    actual = mod.simulate([1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=eps2, initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, eps + eps3)
    mod = structural.UnobservedComponents([0], 'random walk')
    actual = mod.simulate([1.0], nobs, measurement_shocks=eps, state_shocks=eps2, initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, eps + eps3)
    warning = SpecificationWarning
    match = 'irregular component added'
    with pytest.warns(warning, match=match):
        mod = structural.UnobservedComponents([0], 'fixed slope')
    actual = mod.simulate([1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=eps2, initial_state=[0, 1])
    assert_allclose(actual, eps + np.arange(100))
    mod = structural.UnobservedComponents([0], 'deterministic trend')
    actual = mod.simulate([1.0], nobs, measurement_shocks=eps, state_shocks=eps2, initial_state=[0, 1])
    assert_allclose(actual, eps + np.arange(100))
    mod = structural.UnobservedComponents([0], 'local linear deterministic trend')
    actual = mod.simulate([1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=eps2, initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'random walk with drift')
    actual = mod.simulate([1.0], nobs, state_shocks=eps2, initial_state=[0, 1])
    desired = np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'local linear trend')
    actual = mod.simulate([1.0, 1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=np.c_[eps2, eps1], initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)
    actual = mod.simulate([1.0, 1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=np.c_[eps1, eps2], initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'smooth trend')
    actual = mod.simulate([1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=eps1, initial_state=[0, 1])
    desired = eps + np.r_[np.arange(100)]
    assert_allclose(actual, desired)
    actual = mod.simulate([1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=eps2, initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'random trend')
    actual = mod.simulate([1.0, 1.0], nobs, state_shocks=eps1, initial_state=[0, 1])
    desired = np.r_[np.arange(100)]
    assert_allclose(actual, desired)
    actual = mod.simulate([1.0, 1.0], nobs, state_shocks=eps2, initial_state=[0, 1])
    desired = np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2, stochastic_seasonal=False)
    actual = mod.simulate([1.0], nobs, measurement_shocks=eps, initial_state=[10])
    desired = eps + np.tile([10, -10], 50)
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2)
    actual = mod.simulate([1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=eps2, initial_state=[10])
    desired = eps + np.r_[np.tile([10, -10], 25), np.tile([11, -11], 25)]
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True)
    actual = mod.simulate([1.0, 1.2], nobs, measurement_shocks=eps, initial_state=[1, 0])
    x1 = [np.cos(1.2), np.sin(1.2)]
    x2 = [-np.sin(1.2), np.cos(1.2)]
    T = np.array([x1, x2])
    desired = eps
    states = [1, 0]
    for i in range(nobs):
        desired[i] += states[0]
        states = np.dot(T, states)
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True, stochastic_cycle=True)
    actual = mod.simulate([1.0, 1.0, 1.2], nobs, measurement_shocks=eps, state_shocks=np.c_[eps2, eps2], initial_state=[1, 0])
    x1 = [np.cos(1.2), np.sin(1.2)]
    x2 = [-np.sin(1.2), np.cos(1.2)]
    T = np.array([x1, x2])
    desired = eps
    states = [1, 0]
    for i in range(nobs):
        desired[i] += states[0]
        states = np.dot(T, states) + eps2[i]
    assert_allclose(actual, desired)