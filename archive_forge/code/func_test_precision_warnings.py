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
@pytest.mark.parametrize('zero_division', ['warn', 0, 1, np.nan])
def test_precision_warnings(zero_division):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        precision_score(np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]), average='micro', zero_division=zero_division)
        if zero_division == 'warn':
            assert str(record.pop().message) == 'Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.'
        else:
            assert len(record) == 0
        precision_score([0, 0], [0, 0])
        if zero_division == 'warn':
            assert str(record.pop().message) == 'Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.'
    assert_no_warnings(precision_score, np.array([[0, 0], [0, 0]]), np.array([[1, 1], [1, 1]]), average='micro', zero_division=zero_division)