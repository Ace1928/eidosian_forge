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
def test_unique_labels_non_specific():
    for format in ['binary', 'multiclass', 'multilabel-indicator']:
        for y in EXAMPLES[format]:
            unique_labels(y)
    for example in NON_ARRAY_LIKE_EXAMPLES:
        with pytest.raises(ValueError):
            unique_labels(example)
    for y_type in ['unknown', 'continuous', 'continuous-multioutput', 'multiclass-multioutput']:
        for example in EXAMPLES[y_type]:
            with pytest.raises(ValueError):
                unique_labels(example)