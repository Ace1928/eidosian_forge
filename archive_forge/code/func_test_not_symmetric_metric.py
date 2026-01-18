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
@pytest.mark.parametrize('name', sorted(NOT_SYMMETRIC_METRICS))
def test_not_symmetric_metric(name):
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(20,))
    y_pred = random_state.randint(0, 2, size=(20,))
    if name in METRICS_REQUIRE_POSITIVE_Y:
        y_true, y_pred = _require_positive_targets(y_true, y_pred)
    metric = ALL_METRICS[name]
    with pytest.raises(AssertionError):
        assert_array_equal(metric(y_true, y_pred), metric(y_pred, y_true))
        raise ValueError('%s seems to be symmetric' % name)