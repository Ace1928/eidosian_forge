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
@pytest.mark.parametrize('name', sorted(MULTILABELS_METRICS))
def test_raise_value_error_multilabel_sequences(name):
    multilabel_sequences = [[[1], [2], [0, 1]], [(), 2, (0, 1)], [[]], [()], np.array([[], [1, 2]], dtype='object')]
    metric = ALL_METRICS[name]
    for seq in multilabel_sequences:
        with pytest.raises(ValueError):
            metric(seq, seq)