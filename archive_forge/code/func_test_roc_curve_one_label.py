import re
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import (
def test_roc_curve_one_label():
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    expected_message = 'No negative samples in y_true, false positive value should be meaningless'
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    assert_array_equal(fpr, np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape
    expected_message = 'No positive samples in y_true, true positive value should be meaningless'
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        fpr, tpr, thresholds = roc_curve([1 - x for x in y_true], y_pred)
    assert_array_equal(tpr, np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape