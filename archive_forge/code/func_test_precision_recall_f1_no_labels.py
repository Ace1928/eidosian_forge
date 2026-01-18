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
@pytest.mark.parametrize('beta', [1])
@pytest.mark.parametrize('average', ['macro', 'micro', 'weighted', 'samples'])
@pytest.mark.parametrize('zero_division', [0, 1, np.nan])
def test_precision_recall_f1_no_labels(beta, average, zero_division):
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)
    p, r, f, s = assert_no_warnings(precision_recall_fscore_support, y_true, y_pred, average=average, beta=beta, zero_division=zero_division)
    fbeta = assert_no_warnings(fbeta_score, y_true, y_pred, beta=beta, average=average, zero_division=zero_division)
    assert s is None
    if np.isnan(zero_division):
        for metric in [p, r, f, fbeta]:
            assert np.isnan(metric)
        return
    zero_division = float(zero_division)
    assert_almost_equal(p, zero_division)
    assert_almost_equal(r, zero_division)
    assert_almost_equal(f, zero_division)
    assert_almost_equal(fbeta, float(zero_division))