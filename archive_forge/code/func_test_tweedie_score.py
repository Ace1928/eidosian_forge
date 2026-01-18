import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose
from scipy import linalg
from scipy.optimize import minimize, root
from sklearn._loss import HalfBinomialLoss, HalfPoissonLoss, HalfTweedieLoss
from sklearn._loss.link import IdentityLink, LogLink
from sklearn.base import clone
from sklearn.datasets import make_low_rank_matrix, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._glm import _GeneralizedLinearRegressor
from sklearn.linear_model._glm._newton_solver import NewtonCholeskySolver
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance
from sklearn.model_selection import train_test_split
@pytest.mark.parametrize('power', [0, 1, 1.5, 2, 3])
@pytest.mark.parametrize('link', ['log', 'identity'])
def test_tweedie_score(regression_data, power, link):
    """Test that GLM score equals d2_tweedie_score for Tweedie losses."""
    X, y = regression_data
    y = np.abs(y) + 1.0
    glm = TweedieRegressor(power=power, link=link).fit(X, y)
    assert glm.score(X, y) == pytest.approx(d2_tweedie_score(y, glm.predict(X), power=power))