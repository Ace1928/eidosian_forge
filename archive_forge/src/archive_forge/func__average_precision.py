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
def _average_precision(y_true, y_score):
    """Alternative implementation to check for correctness of
    `average_precision_score`.

    Note that this implementation fails on some edge cases.
    For example, for constant predictions e.g. [0.5, 0.5, 0.5],
    y_true = [1, 0, 0] returns an average precision of 0.33...
    but y_true = [0, 0, 1] returns 1.0.
    """
    pos_label = np.unique(y_true)[1]
    n_pos = np.sum(y_true == pos_label)
    order = np.argsort(y_score)[::-1]
    y_score = y_score[order]
    y_true = y_true[order]
    score = 0
    for i in range(len(y_score)):
        if y_true[i] == pos_label:
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= i + 1.0
            score += prec
    return score / n_pos