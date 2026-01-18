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
def test_nonfinite_params():
    rng = np.random.RandomState(0)
    n_samples = 10
    fmax = np.finfo(np.float64).max
    X = fmax * rng.uniform(size=(n_samples, 2))
    y = rng.standard_normal(size=n_samples)
    clf = MLPRegressor()
    msg = 'Solver produced non-finite parameter weights. The input data may contain large values and need to be preprocessed.'
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)