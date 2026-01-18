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
@pytest.mark.parametrize('y_true, y_pred', [([0], [0]), ([], [])])
@pytest.mark.parametrize('metric', [f1_score, partial(fbeta_score, beta=1), precision_score, recall_score])
def test_zero_division_nan_warning(metric, y_true, y_pred):
    """Check the behaviour of `zero_division` when setting to "warn".
    A `UndefinedMetricWarning` should be raised.
    """
    with pytest.warns(UndefinedMetricWarning):
        result = metric(y_true, y_pred, zero_division='warn')
    assert result == 0.0