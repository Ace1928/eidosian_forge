import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_unreachable_accuracy():
    assert_array_almost_equal(orthogonal_mp(X, y, tol=0), orthogonal_mp(X, y, n_nonzero_coefs=n_features))
    warning_message = 'Orthogonal matching pursuit ended prematurely due to linear dependence in the dictionary. The requested precision might not have been met.'
    with pytest.warns(RuntimeWarning, match=warning_message):
        assert_array_almost_equal(orthogonal_mp(X, y, tol=0, precompute=True), orthogonal_mp(X, y, precompute=True, n_nonzero_coefs=n_features))