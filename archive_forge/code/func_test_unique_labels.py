from itertools import product
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import config_context, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import (
def test_unique_labels():
    with pytest.raises(ValueError):
        unique_labels()
    assert_array_equal(unique_labels(range(10)), np.arange(10))
    assert_array_equal(unique_labels(np.arange(10)), np.arange(10))
    assert_array_equal(unique_labels([4, 0, 2]), np.array([0, 2, 4]))
    assert_array_equal(unique_labels(np.array([[0, 0, 1], [1, 0, 1], [0, 0, 0]])), np.arange(3))
    assert_array_equal(unique_labels(np.array([[0, 0, 1], [0, 0, 0]])), np.arange(3))
    assert_array_equal(unique_labels([4, 0, 2], range(5)), np.arange(5))
    assert_array_equal(unique_labels((0, 1, 2), (0,), (2, 1)), np.arange(3))
    with pytest.raises(ValueError):
        unique_labels([4, 0, 2], np.ones((5, 5)))
    with pytest.raises(ValueError):
        unique_labels(np.ones((5, 4)), np.ones((5, 5)))
    assert_array_equal(unique_labels(np.ones((4, 5)), np.ones((5, 5))), np.arange(5))