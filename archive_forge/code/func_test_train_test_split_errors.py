import re
import warnings
from itertools import combinations, combinations_with_replacement, permutations
import numpy as np
import pytest
from scipy import stats
from scipy.sparse import issparse
from scipy.special import comb
from sklearn import config_context
from sklearn.datasets import load_digits, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
from sklearn.model_selection._split import (
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import assert_request_is_empty
from sklearn.utils._array_api import (
from sklearn.utils._array_api import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_train_test_split_errors():
    pytest.raises(ValueError, train_test_split)
    pytest.raises(ValueError, train_test_split, range(3), train_size=1.1)
    pytest.raises(ValueError, train_test_split, range(3), test_size=0.6, train_size=0.6)
    pytest.raises(ValueError, train_test_split, range(3), test_size=np.float32(0.6), train_size=np.float32(0.6))
    pytest.raises(ValueError, train_test_split, range(3), test_size='wrong_type')
    pytest.raises(ValueError, train_test_split, range(3), test_size=2, train_size=4)
    pytest.raises(TypeError, train_test_split, range(3), some_argument=1.1)
    pytest.raises(ValueError, train_test_split, range(3), range(42))
    pytest.raises(ValueError, train_test_split, range(10), shuffle=False, stratify=True)
    with pytest.raises(ValueError, match='train_size=11 should be either positive and smaller than the number of samples 10 or a float in the \\(0, 1\\) range'):
        train_test_split(range(10), train_size=11, test_size=1)