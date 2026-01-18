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
@pytest.mark.parametrize('drop', [True, False])
def test_roc_curve(drop):
    y_true, _, y_score = make_prediction(binary=True)
    expected_auc = _auc(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=drop)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, expected_auc, decimal=2)
    assert_almost_equal(roc_auc, roc_auc_score(y_true, y_score))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape