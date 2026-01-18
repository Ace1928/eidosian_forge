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
def test_precision_recall_f_extra_labels():
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]
    for i, (y_true, y_pred) in enumerate(data):
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None)
        assert_array_almost_equal([0.0, 1.0, 1.0, 0.5, 0.0], actual)
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro')
        assert_array_almost_equal(np.mean([0.0, 1.0, 1.0, 0.5, 0.0]), actual)
        for average in ['micro', 'weighted', 'samples']:
            if average == 'samples' and i == 0:
                continue
            assert_almost_equal(recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=average), recall_score(y_true, y_pred, labels=None, average=average))
    for average in [None, 'macro', 'micro', 'samples']:
        with pytest.raises(ValueError):
            recall_score(y_true_bin, y_pred_bin, labels=np.arange(6), average=average)
        with pytest.raises(ValueError):
            recall_score(y_true_bin, y_pred_bin, labels=np.arange(-1, 4), average=average)
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='samples', labels=[0, 1])
    assert_almost_equal(np.array([p, r, f]), np.array([3 / 4, 1, 5 / 6]))