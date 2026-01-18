from numpy.testing import (
from numpy import (
import numpy as np
import pytest
def test_tril_triu_with_inf():
    arr = np.array([[1, 1, np.inf], [1, 1, 1], [np.inf, 1, 1]])
    out_tril = np.array([[1, 0, 0], [1, 1, 0], [np.inf, 1, 1]])
    out_triu = out_tril.T
    assert_array_equal(np.triu(arr), out_triu)
    assert_array_equal(np.tril(arr), out_tril)