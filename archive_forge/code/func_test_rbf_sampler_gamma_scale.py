import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_rbf_sampler_gamma_scale():
    """Check the inner value computed when `gamma='scale'`."""
    X, y = ([[0.0], [1.0]], [0, 1])
    rbf = RBFSampler(gamma='scale')
    rbf.fit(X, y)
    assert rbf._gamma == pytest.approx(4)