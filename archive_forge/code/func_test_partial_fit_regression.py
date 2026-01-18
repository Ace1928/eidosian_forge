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
def test_partial_fit_regression():
    X = X_reg
    y = y_reg
    for momentum in [0, 0.9]:
        mlp = MLPRegressor(solver='sgd', max_iter=100, activation='relu', random_state=1, learning_rate_init=0.01, batch_size=X.shape[0], momentum=momentum)
        with warnings.catch_warnings(record=True):
            mlp.fit(X, y)
        pred1 = mlp.predict(X)
        mlp = MLPRegressor(solver='sgd', activation='relu', learning_rate_init=0.01, random_state=1, batch_size=X.shape[0], momentum=momentum)
        for i in range(100):
            mlp.partial_fit(X, y)
        pred2 = mlp.predict(X)
        assert_allclose(pred1, pred2)
        score = mlp.score(X, y)
        assert score > 0.65