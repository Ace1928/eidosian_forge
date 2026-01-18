import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('expected_outliers', [30, 53])
def test_predicted_outlier_number(expected_outliers):
    X = iris.data
    n_samples = X.shape[0]
    contamination = float(expected_outliers) / n_samples
    clf = neighbors.LocalOutlierFactor(contamination=contamination)
    y_pred = clf.fit_predict(X)
    num_outliers = np.sum(y_pred != 1)
    if num_outliers != expected_outliers:
        y_dec = clf.negative_outlier_factor_
        check_outlier_corruption(num_outliers, expected_outliers, y_dec)