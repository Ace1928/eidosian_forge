import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_with_without_gram_tol():
    assert_array_almost_equal(orthogonal_mp(X, y, tol=1.0), orthogonal_mp(X, y, tol=1.0, precompute=True))