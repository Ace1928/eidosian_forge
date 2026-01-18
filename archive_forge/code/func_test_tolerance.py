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
def test_tolerance():
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = MLPClassifier(tol=0.5, max_iter=3000, solver='sgd')
    clf.fit(X, y)
    assert clf.max_iter > clf.n_iter_