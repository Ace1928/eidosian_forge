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
def test_precision_recall_f_ignored_labels():
    y_true = [1, 1, 2, 3]
    y_pred = [1, 3, 3, 3]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]
    for i, (y_true, y_pred) in enumerate(data):
        recall_13 = partial(recall_score, y_true, y_pred, labels=[1, 3])
        recall_all = partial(recall_score, y_true, y_pred, labels=None)
        assert_array_almost_equal([0.5, 1.0], recall_13(average=None))
        assert_almost_equal((0.5 + 1.0) / 2, recall_13(average='macro'))
        assert_almost_equal((0.5 * 2 + 1.0 * 1) / 3, recall_13(average='weighted'))
        assert_almost_equal(2.0 / 3, recall_13(average='micro'))
        for average in ['macro', 'weighted', 'micro']:
            assert recall_13(average=average) != recall_all(average=average)