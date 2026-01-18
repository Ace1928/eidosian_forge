from functools import partial
from inspect import signature
from itertools import chain, permutations, product
import numpy as np
import pytest
from sklearn._config import config_context
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import (
from sklearn.metrics._base import _average_binary_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_random_state
@pytest.mark.parametrize('name', sorted(set(THRESHOLDED_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS))
def test_thresholded_metric_permutation_invariance(name):
    n_samples, n_classes = (100, 3)
    random_state = check_random_state(0)
    y_score = random_state.rand(n_samples, n_classes)
    temp = np.exp(-y_score)
    y_score = temp / temp.sum(axis=-1).reshape(-1, 1)
    y_true = random_state.randint(0, n_classes, size=n_samples)
    metric = ALL_METRICS[name]
    score = metric(y_true, y_score)
    for perm in permutations(range(n_classes), n_classes):
        inverse_perm = np.zeros(n_classes, dtype=int)
        inverse_perm[list(perm)] = np.arange(n_classes)
        y_score_perm = y_score[:, inverse_perm]
        y_true_perm = np.take(perm, y_true)
        current_score = metric(y_true_perm, y_score_perm)
        assert_almost_equal(score, current_score)