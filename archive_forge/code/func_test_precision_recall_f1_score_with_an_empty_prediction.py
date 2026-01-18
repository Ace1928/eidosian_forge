import re
import warnings
from functools import partial
from itertools import chain, permutations, product
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._classification import _check_targets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
@ignore_warnings
@pytest.mark.parametrize('zero_division, zero_division_expected', [('warn', 0), (0, 0), (1, 1), (np.nan, np.nan)])
def test_precision_recall_f1_score_with_an_empty_prediction(zero_division, zero_division_expected):
    y_true = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=zero_division)
    assert_array_almost_equal(p, [zero_division_expected, 1.0, 1.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 0.5, 1.0, zero_division_expected], 2)
    expected_f = 0
    assert_array_almost_equal(f, [expected_f, 1 / 1.5, 1, expected_f], 2)
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)
    f2 = fbeta_score(y_true, y_pred, beta=2, average=None, zero_division=zero_division)
    support = s
    assert_array_almost_equal(f2, [expected_f, 0.55, 1, expected_f], 2)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=zero_division)
    value_to_sum = 0 if np.isnan(zero_division_expected) else zero_division_expected
    values_to_average = 3 + (not np.isnan(zero_division_expected))
    assert_almost_equal(p, (2 + value_to_sum) / values_to_average)
    assert_almost_equal(r, (1.5 + value_to_sum) / values_to_average)
    expected_f = (2 / 3 + 1) / 4
    assert_almost_equal(f, expected_f)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=zero_division), _nanaverage(f2, weights=None))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=zero_division)
    assert_almost_equal(p, 2 / 3)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2 / 3 / (2 / 3 + 0.5))
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average='micro', zero_division=zero_division), (1 + 4) * p * r / (4 * p + r))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=zero_division)
    assert_almost_equal(p, 3 / 4 if zero_division_expected == 0 else 1.0)
    assert_almost_equal(r, 0.5)
    values_to_average = 4
    assert_almost_equal(f, (2 * 2 / 3 + 1) / values_to_average)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average='weighted', zero_division=zero_division), _nanaverage(f2, weights=support))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='samples')
    assert_almost_equal(p, 1 / 3)
    assert_almost_equal(r, 1 / 3)
    assert_almost_equal(f, 1 / 3)
    assert s is None
    expected_result = 0.333
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average='samples', zero_division=zero_division), expected_result, 2)