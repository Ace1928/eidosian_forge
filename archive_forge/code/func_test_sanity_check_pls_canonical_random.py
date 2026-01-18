import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.cross_decomposition._pls import (
from sklearn.datasets import load_linnerud, make_regression
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
def test_sanity_check_pls_canonical_random():
    n = 500
    p_noise = 10
    q_noise = 5
    rng = check_random_state(11)
    l1 = rng.normal(size=n)
    l2 = rng.normal(size=n)
    latents = np.array([l1, l1, l2, l2]).T
    X = latents + rng.normal(size=4 * n).reshape((n, 4))
    Y = latents + rng.normal(size=4 * n).reshape((n, 4))
    X = np.concatenate((X, rng.normal(size=p_noise * n).reshape(n, p_noise)), axis=1)
    Y = np.concatenate((Y, rng.normal(size=q_noise * n).reshape(n, q_noise)), axis=1)
    pls = PLSCanonical(n_components=3)
    pls.fit(X, Y)
    expected_x_weights = np.array([[0.65803719, 0.19197924, 0.21769083], [0.7009113, 0.13303969, -0.15376699], [0.13528197, -0.68636408, 0.13856546], [0.16854574, -0.66788088, -0.12485304], [-0.03232333, -0.04189855, 0.40690153], [0.1148816, -0.09643158, 0.1613305], [0.04792138, -0.02384992, 0.17175319], [-0.06781, -0.01666137, -0.18556747], [-0.00266945, -0.00160224, 0.11893098], [-0.00849528, -0.07706095, 0.1570547], [-0.00949471, -0.02964127, 0.34657036], [-0.03572177, 0.0945091, 0.3414855], [0.05584937, -0.02028961, -0.57682568], [0.05744254, -0.01482333, -0.17431274]])
    expected_x_loadings = np.array([[0.65649254, 0.1847647, 0.15270699], [0.67554234, 0.15237508, -0.09182247], [0.19219925, -0.67750975, 0.08673128], [0.2133631, -0.67034809, -0.08835483], [-0.03178912, -0.06668336, 0.43395268], [0.15684588, -0.13350241, 0.20578984], [0.03337736, -0.03807306, 0.09871553], [-0.06199844, 0.01559854, -0.1881785], [0.00406146, -0.00587025, 0.16413253], [-0.00374239, -0.05848466, 0.19140336], [0.00139214, -0.01033161, 0.32239136], [-0.05292828, 0.0953533, 0.31916881], [0.04031924, -0.01961045, -0.65174036], [0.06172484, -0.06597366, -0.1244497]])
    expected_y_weights = np.array([[0.66101097, 0.18672553, 0.22826092], [0.69347861, 0.18463471, -0.23995597], [0.14462724, -0.66504085, 0.17082434], [0.22247955, -0.6932605, -0.09832993], [0.07035859, 0.00714283, 0.67810124], [0.07765351, -0.0105204, -0.44108074], [-0.00917056, 0.04322147, 0.10062478], [-0.01909512, 0.06182718, 0.28830475], [0.01756709, 0.04797666, 0.32225745]])
    expected_y_loadings = np.array([[0.68568625, 0.1674376, 0.0969508], [0.68782064, 0.20375837, -0.1164448], [0.11712173, -0.68046903, 0.12001505], [0.17860457, -0.6798319, -0.05089681], [0.06265739, -0.0277703, 0.74729584], [0.0914178, 0.00403751, -0.5135078], [-0.02196918, -0.01377169, 0.09564505], [-0.03288952, 0.09039729, 0.31858973], [0.04287624, 0.05254676, 0.27836841]])
    assert_array_almost_equal(np.abs(pls.x_loadings_), np.abs(expected_x_loadings))
    assert_array_almost_equal(np.abs(pls.x_weights_), np.abs(expected_x_weights))
    assert_array_almost_equal(np.abs(pls.y_loadings_), np.abs(expected_y_loadings))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_weights))
    x_loadings_sign_flip = np.sign(pls.x_loadings_ / expected_x_loadings)
    x_weights_sign_flip = np.sign(pls.x_weights_ / expected_x_weights)
    y_weights_sign_flip = np.sign(pls.y_weights_ / expected_y_weights)
    y_loadings_sign_flip = np.sign(pls.y_loadings_ / expected_y_loadings)
    assert_array_almost_equal(x_loadings_sign_flip, x_weights_sign_flip)
    assert_array_almost_equal(y_loadings_sign_flip, y_weights_sign_flip)
    assert_matrix_orthogonal(pls.x_weights_)
    assert_matrix_orthogonal(pls.y_weights_)
    assert_matrix_orthogonal(pls._x_scores)
    assert_matrix_orthogonal(pls._y_scores)