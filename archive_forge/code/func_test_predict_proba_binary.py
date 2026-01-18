import re
import sys
import warnings
from io import StringIO
import joblib
import numpy as np
import pytest
from numpy.testing import (
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, scale
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS
def test_predict_proba_binary():
    X = X_digits_binary[:50]
    y = y_digits_binary[:50]
    clf = MLPClassifier(hidden_layer_sizes=5, activation='logistic', random_state=1)
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
    y_proba = clf.predict_proba(X)
    y_log_proba = clf.predict_log_proba(X)
    n_samples, n_classes = (y.shape[0], 2)
    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)
    assert y_proba.shape == (n_samples, n_classes)
    assert_array_equal(proba_max, proba_log_max)
    assert_allclose(y_log_proba, np.log(y_proba))
    assert roc_auc_score(y, y_proba[:, 1]) == 1.0