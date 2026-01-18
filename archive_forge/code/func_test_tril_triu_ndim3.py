from numpy.testing import (
from numpy import (
import numpy as np
import pytest
def test_tril_triu_ndim3():
    for dtype in np.typecodes['AllFloat'] + np.typecodes['AllInteger']:
        a = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=dtype)
        a_tril_desired = np.array([[[1, 0], [1, 1]], [[1, 0], [1, 0]], [[1, 0], [0, 0]]], dtype=dtype)
        a_triu_desired = np.array([[[1, 1], [0, 1]], [[1, 1], [0, 0]], [[1, 1], [0, 0]]], dtype=dtype)
        a_triu_observed = np.triu(a)
        a_tril_observed = np.tril(a)
        assert_array_equal(a_triu_observed, a_triu_desired)
        assert_array_equal(a_tril_observed, a_tril_desired)
        assert_equal(a_triu_observed.dtype, a.dtype)
        assert_equal(a_tril_observed.dtype, a.dtype)