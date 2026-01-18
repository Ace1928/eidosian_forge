import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tsa.regime_switching import markov_switching
def test_transition_matrix():
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2)
    params = np.r_[0.0, 0.0, 1.0]
    transition_matrix = np.zeros((2, 2, 1))
    transition_matrix[1, :] = 1.0
    assert_allclose(mod.regime_transition_matrix(params), transition_matrix)
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=3)
    params = np.r_[[0] * 3, [0.2] * 3, 1.0]
    transition_matrix = np.zeros((3, 3, 1))
    transition_matrix[1, :, 0] = 0.2
    transition_matrix[2, :, 0] = 0.8
    assert_allclose(mod.regime_transition_matrix(params), transition_matrix)
    endog = np.ones(10)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2, exog_tvtp=exog_tvtp)
    params = np.r_[0, 0, 0, 0]
    assert_allclose(mod.regime_transition_matrix(params), 0.5)
    params = np.r_[1, 2, 1, 2]
    transition_matrix = np.zeros((2, 2, 10))
    coeffs0 = np.sum(exog_tvtp, axis=1)
    p11 = np.exp(coeffs0) / (1 + np.exp(coeffs0))
    transition_matrix[0, 0, :] = p11
    transition_matrix[1, 0, :] = 1 - p11
    coeffs1 = np.sum(2 * exog_tvtp, axis=1)
    p21 = np.exp(coeffs1) / (1 + np.exp(coeffs1))
    transition_matrix[0, 1, :] = p21
    transition_matrix[1, 1, :] = 1 - p21
    assert_allclose(mod.regime_transition_matrix(params), transition_matrix, atol=1e-10)
    endog = np.ones(10)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    mod = markov_switching.MarkovSwitching(endog, k_regimes=3, exog_tvtp=exog_tvtp)
    params = np.r_[[0] * 12]
    assert_allclose(mod.regime_transition_matrix(params), 1 / 3)
    params = np.r_[[0] * 6, [2] * 6]
    transition_matrix = np.zeros((3, 3, 10))
    p11 = np.zeros(10)
    p12 = 2 * np.sum(exog_tvtp, axis=1)
    tmp = np.exp(np.c_[p11, p12]).T
    transition_matrix[:2, 0, :] = tmp / (1 + np.sum(tmp, axis=0))
    transition_matrix[2, 0, :] = 1 - np.sum(transition_matrix[:2, 0, :], axis=0)
    assert_allclose(mod.regime_transition_matrix(params)[:, 0, :], transition_matrix[:, 0, :], atol=1e-10)