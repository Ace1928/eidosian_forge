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
def test_symmetry_consistency():
    assert SYMMETRIC_METRICS | NOT_SYMMETRIC_METRICS | set(THRESHOLDED_METRICS) | METRIC_UNDEFINED_BINARY_MULTICLASS == set(ALL_METRICS)
    assert SYMMETRIC_METRICS & NOT_SYMMETRIC_METRICS == set()