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
@pytest.mark.parametrize('y_true, y_score, labels, msg', [([0, 0.57, 1, 2], [[0.2, 0.1, 0.7], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.4, 0.5, 0.1]], None, "y type must be 'binary' or 'multiclass', got 'continuous'"), ([0, 1, 2, 3], [[0.2, 0.1, 0.7], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.4, 0.5, 0.1]], None, "Number of classes in 'y_true' \\(4\\) not equal to the number of classes in 'y_score' \\(3\\)."), (['c', 'c', 'a', 'b'], [[0.2, 0.1, 0.7], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.4, 0.5, 0.1]], ['a', 'b', 'c', 'c'], "Parameter 'labels' must be unique."), (['c', 'c', 'a', 'b'], [[0.2, 0.1, 0.7], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.4, 0.5, 0.1]], ['a', 'c', 'b'], "Parameter 'labels' must be ordered."), ([0, 0, 1, 2], [[0.2, 0.1, 0.7], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.4, 0.5, 0.1]], [0, 1, 2, 3], "Number of given labels \\(4\\) not equal to the number of classes in 'y_score' \\(3\\)."), ([0, 0, 1, 2], [[0.2, 0.1, 0.7], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.4, 0.5, 0.1]], [0, 1, 3], "'y_true' contains labels not in parameter 'labels'."), ([0, 1], [[0.5, 0.2, 0.2], [0.3, 0.4, 0.2]], None, '`y_true` is binary while y_score is 2d with 3 classes. If `y_true` does not contain all the labels, `labels` must be provided')])
def test_top_k_accuracy_score_error(y_true, y_score, labels, msg):
    with pytest.raises(ValueError, match=msg):
        top_k_accuracy_score(y_true, y_score, k=2, labels=labels)