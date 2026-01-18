import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
@pytest.mark.parametrize('Estimator', [linear_model.SGDClassifier, linear_model.SGDRegressor])
def test_sgd_random_state(Estimator, global_random_seed):
    if Estimator == linear_model.SGDRegressor:
        X, y = datasets.make_regression(random_state=global_random_seed)
    else:
        X, y = datasets.make_classification(random_state=global_random_seed)
    est = Estimator(random_state=global_random_seed, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_same_seed_a = est.fit(X, y).coef_
        assert est.n_iter_ == 1
    est = Estimator(random_state=global_random_seed, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_same_seed_b = est.fit(X, y).coef_
        assert est.n_iter_ == 1
    assert_allclose(coef_same_seed_a, coef_same_seed_b)
    est = Estimator(random_state=global_random_seed + 1, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_other_seed = est.fit(X, y).coef_
        assert est.n_iter_ == 1
    assert np.abs(coef_same_seed_a - coef_other_seed).max() > 1.0