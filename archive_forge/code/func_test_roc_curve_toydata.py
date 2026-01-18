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
def test_roc_curve_toydata():
    y_true = [0, 1]
    y_score = [0, 1]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)
    y_true = [0, 1]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1, 1])
    assert_array_almost_equal(fpr, [0, 0, 1])
    assert_almost_equal(roc_auc, 0.0)
    y_true = [1, 0]
    y_score = [1, 1]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)
    y_true = [1, 0]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)
    y_true = [1, 0]
    y_score = [0.5, 0.5]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)
    y_true = [0, 0]
    y_score = [0.25, 0.75]
    expected_message = 'No positive samples in y_true, true positive value should be meaningless'
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0.0, 0.5, 1.0])
    assert_array_almost_equal(fpr, [np.nan, np.nan, np.nan])
    y_true = [1, 1]
    y_score = [0.25, 0.75]
    expected_message = 'No negative samples in y_true, false positive value should be meaningless'
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [np.nan, np.nan, np.nan])
    assert_array_almost_equal(fpr, [0.0, 0.5, 1.0])
    y_true = np.array([[0, 1], [0, 1]])
    y_score = np.array([[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average='macro')
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average='weighted')
    assert_almost_equal(roc_auc_score(y_true, y_score, average='samples'), 1.0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average='micro'), 1.0)
    y_true = np.array([[0, 1], [0, 1]])
    y_score = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average='macro')
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average='weighted')
    assert_almost_equal(roc_auc_score(y_true, y_score, average='samples'), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average='micro'), 0.5)
    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0, 1], [1, 0]])
    assert_almost_equal(roc_auc_score(y_true, y_score, average='macro'), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average='weighted'), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average='samples'), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average='micro'), 0)
    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert_almost_equal(roc_auc_score(y_true, y_score, average='macro'), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average='weighted'), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average='samples'), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average='micro'), 0.5)