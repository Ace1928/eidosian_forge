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
def test_partial_fit_classification():
    for X, y in classification_datasets:
        mlp = MLPClassifier(solver='sgd', max_iter=100, random_state=1, tol=0, alpha=1e-05, learning_rate_init=0.2)
        with ignore_warnings(category=ConvergenceWarning):
            mlp.fit(X, y)
        pred1 = mlp.predict(X)
        mlp = MLPClassifier(solver='sgd', random_state=1, alpha=1e-05, learning_rate_init=0.2)
        for i in range(100):
            mlp.partial_fit(X, y, classes=np.unique(y))
        pred2 = mlp.predict(X)
        assert_array_equal(pred1, pred2)
        assert mlp.score(X, y) > 0.95