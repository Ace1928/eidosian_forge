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
@pytest.mark.parametrize('metric', chain(THRESHOLDED_METRICS.values(), REGRESSION_METRICS.values()))
@pytest.mark.parametrize('y_true, y_score', invalids_nan_inf)
def test_regression_thresholded_inf_nan_input(metric, y_true, y_score):
    if metric == coverage_error:
        y_true = [y_true]
        y_score = [y_score]
    with pytest.raises(ValueError, match='contains (NaN|infinity)'):
        metric(y_true, y_score)