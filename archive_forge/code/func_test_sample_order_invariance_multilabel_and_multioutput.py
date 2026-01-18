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
@ignore_warnings
def test_sample_order_invariance_multilabel_and_multioutput():
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(20, 25))
    y_pred = random_state.randint(0, 2, size=(20, 25))
    y_score = random_state.normal(size=y_true.shape)
    y_true_shuffle, y_pred_shuffle, y_score_shuffle = shuffle(y_true, y_pred, y_score, random_state=0)
    for name in MULTILABELS_METRICS:
        metric = ALL_METRICS[name]
        assert_allclose(metric(y_true, y_pred), metric(y_true_shuffle, y_pred_shuffle), err_msg='%s is not sample order invariant' % name)
    for name in THRESHOLDED_MULTILABEL_METRICS:
        metric = ALL_METRICS[name]
        assert_allclose(metric(y_true, y_score), metric(y_true_shuffle, y_score_shuffle), err_msg='%s is not sample order invariant' % name)
    for name in MULTIOUTPUT_METRICS:
        metric = ALL_METRICS[name]
        assert_allclose(metric(y_true, y_score), metric(y_true_shuffle, y_score_shuffle), err_msg='%s is not sample order invariant' % name)
        assert_allclose(metric(y_true, y_pred), metric(y_true_shuffle, y_pred_shuffle), err_msg='%s is not sample order invariant' % name)