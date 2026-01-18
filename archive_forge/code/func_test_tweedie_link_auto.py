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
@pytest.mark.parametrize('power, expected_link_class', [(0, IdentityLink), (1, LogLink), (2, LogLink), (3, LogLink)])
def test_tweedie_link_auto(power, expected_link_class):
    """Test that link='auto' delivers the expected link function"""
    y = np.array([0.1, 0.5])
    X = np.array([[1], [2]])
    glm = TweedieRegressor(link='auto', power=power).fit(X, y)
    assert isinstance(glm._base_loss.link, expected_link_class)