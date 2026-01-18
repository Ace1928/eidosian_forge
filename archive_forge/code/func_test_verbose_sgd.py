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
def test_verbose_sgd():
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = MLPClassifier(solver='sgd', max_iter=2, verbose=10, hidden_layer_sizes=2)
    old_stdout = sys.stdout
    sys.stdout = output = StringIO()
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
    clf.partial_fit(X, y)
    sys.stdout = old_stdout
    assert 'Iteration' in output.getvalue()