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
def test_score_scale_invariance():
    y_true, _, y_score = make_prediction(binary=True)
    roc_auc = roc_auc_score(y_true, y_score)
    roc_auc_scaled_up = roc_auc_score(y_true, 100 * y_score)
    roc_auc_scaled_down = roc_auc_score(y_true, 1e-06 * y_score)
    roc_auc_shifted = roc_auc_score(y_true, y_score - 10)
    assert roc_auc == roc_auc_scaled_up
    assert roc_auc == roc_auc_scaled_down
    assert roc_auc == roc_auc_shifted
    pr_auc = average_precision_score(y_true, y_score)
    pr_auc_scaled_up = average_precision_score(y_true, 100 * y_score)
    pr_auc_scaled_down = average_precision_score(y_true, 1e-06 * y_score)
    pr_auc_shifted = average_precision_score(y_true, y_score - 10)
    assert pr_auc == pr_auc_scaled_up
    assert pr_auc == pr_auc_scaled_down
    assert pr_auc == pr_auc_shifted