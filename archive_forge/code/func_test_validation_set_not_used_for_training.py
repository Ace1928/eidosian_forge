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
@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_validation_set_not_used_for_training(klass):
    X, Y = (iris.data, iris.target)
    validation_fraction = 0.4
    seed = 42
    shuffle = False
    max_iter = 10
    clf1 = klass(early_stopping=True, random_state=np.random.RandomState(seed), validation_fraction=validation_fraction, learning_rate='constant', eta0=0.01, tol=None, max_iter=max_iter, shuffle=shuffle)
    clf1.fit(X, Y)
    assert clf1.n_iter_ == max_iter
    clf2 = klass(early_stopping=False, random_state=np.random.RandomState(seed), learning_rate='constant', eta0=0.01, tol=None, max_iter=max_iter, shuffle=shuffle)
    if is_classifier(clf2):
        cv = StratifiedShuffleSplit(test_size=validation_fraction, random_state=seed)
    else:
        cv = ShuffleSplit(test_size=validation_fraction, random_state=seed)
    idx_train, idx_val = next(cv.split(X, Y))
    idx_train = np.sort(idx_train)
    clf2.fit(X[idx_train], Y[idx_train])
    assert clf2.n_iter_ == max_iter
    assert_array_equal(clf1.coef_, clf2.coef_)