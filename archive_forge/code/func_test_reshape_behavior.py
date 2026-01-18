from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
def test_reshape_behavior():
    """Check reshape behavior with copy and is strict with non-tuple shape."""
    xp = _NumPyAPIWrapper()
    X = xp.asarray([[1, 2, 3], [3, 4, 5]])
    X_no_copy = xp.reshape(X, (-1,), copy=False)
    assert X_no_copy.base is X
    X_copy = xp.reshape(X, (6, 1), copy=True)
    assert X_copy.base is not X.base
    with pytest.raises(TypeError, match='shape must be a tuple'):
        xp.reshape(X, -1)