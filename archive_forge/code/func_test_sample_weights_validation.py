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
def test_sample_weights_validation():
    """Test the raised errors in the validation of sample_weight."""
    X = [[1]]
    y = [1]
    weights = 0
    glm = _GeneralizedLinearRegressor()
    glm.fit(X, y, sample_weight=1)
    weights = [[0]]
    with pytest.raises(ValueError, match='must be 1D array or scalar'):
        glm.fit(X, y, weights)
    weights = [1, 0]
    msg = 'sample_weight.shape == \\(2,\\), expected \\(1,\\)!'
    with pytest.raises(ValueError, match=msg):
        glm.fit(X, y, weights)