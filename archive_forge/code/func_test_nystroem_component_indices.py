import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_nystroem_component_indices():
    """Check that `component_indices_` corresponds to the subset of
    training points used to construct the feature map.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20474
    """
    X, _ = make_classification(n_samples=100, n_features=20)
    feature_map_nystroem = Nystroem(n_components=10, random_state=0)
    feature_map_nystroem.fit(X)
    assert feature_map_nystroem.component_indices_.shape == (10,)