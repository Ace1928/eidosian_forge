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
@pytest.mark.parametrize('name', sorted(METRICS_WITH_NORMALIZE_OPTION.intersection(MULTILABELS_METRICS)))
def test_normalize_option_multilabel_classification(name):
    n_classes = 4
    n_samples = 100
    random_state = check_random_state(0)
    _, y_true = make_multilabel_classification(n_features=1, n_classes=n_classes, random_state=0, allow_unlabeled=True, n_samples=n_samples)
    _, y_pred = make_multilabel_classification(n_features=1, n_classes=n_classes, random_state=1, allow_unlabeled=True, n_samples=n_samples)
    y_score = random_state.uniform(size=y_true.shape)
    y_true += [0] * n_classes
    y_pred += [0] * n_classes
    metrics = ALL_METRICS[name]
    pred = y_score if name in THRESHOLDED_METRICS else y_pred
    measure_normalized = metrics(y_true, pred, normalize=True)
    measure_not_normalized = metrics(y_true, pred, normalize=False)
    assert_array_less(-1.0 * measure_normalized, 0, err_msg='We failed to test correctly the normalize option')
    assert_allclose(measure_normalized, measure_not_normalized / n_samples, err_msg=f'Failed with {name}')