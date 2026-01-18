import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_subscript_range(self):
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    np.einsum(a, [0, 20], b, [20, 2], [0, 2], optimize=False)
    np.einsum(a, [0, 27], b, [27, 2], [0, 2], optimize=False)
    np.einsum(a, [0, 51], b, [51, 2], [0, 2], optimize=False)
    assert_raises(ValueError, lambda: np.einsum(a, [0, 52], b, [52, 2], [0, 2], optimize=False))
    assert_raises(ValueError, lambda: np.einsum(a, [-1, 5], b, [5, 2], [-1, 2], optimize=False))