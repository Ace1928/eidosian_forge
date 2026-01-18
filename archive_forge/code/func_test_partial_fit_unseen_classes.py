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
def test_partial_fit_unseen_classes():
    clf = MLPClassifier(random_state=0)
    clf.partial_fit([[1], [2], [3]], ['a', 'b', 'c'], classes=['a', 'b', 'c', 'd'])
    clf.partial_fit([[4]], ['d'])
    assert clf.score([[1], [2], [3], [4]], ['a', 'b', 'c', 'd']) > 0