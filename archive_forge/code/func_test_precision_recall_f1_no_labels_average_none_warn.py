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
def test_precision_recall_f1_no_labels_average_none_warn():
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)
    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, beta=1)
    assert_array_almost_equal(p, [0, 0, 0], 2)
    assert_array_almost_equal(r, [0, 0, 0], 2)
    assert_array_almost_equal(f, [0, 0, 0], 2)
    assert_array_almost_equal(s, [0, 0, 0], 2)
    with pytest.warns(UndefinedMetricWarning):
        fbeta = fbeta_score(y_true, y_pred, beta=1, average=None)
    assert_array_almost_equal(fbeta, [0, 0, 0], 2)