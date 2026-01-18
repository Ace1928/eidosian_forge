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
@pytest.mark.parametrize('average', ['samples', 'micro', 'macro', 'weighted', None])
def test_precision_refcall_f1_score_multilabel_unordered_labels(average):
    y_true = np.array([[1, 1, 0, 0]])
    y_pred = np.array([[0, 0, 1, 1]])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[3, 0, 1, 2], warn_for=[], average=average)
    assert_array_equal(p, 0)
    assert_array_equal(r, 0)
    assert_array_equal(f, 0)
    if average is None:
        assert_array_equal(s, [0, 1, 1, 0])