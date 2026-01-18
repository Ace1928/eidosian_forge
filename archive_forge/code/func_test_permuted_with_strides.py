import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_permuted_with_strides(self):
    random = Generator(MT19937(self.seed))
    x0 = np.arange(22).reshape(2, 11)
    x1 = x0.copy()
    x = x0[:, ::3]
    y = random.permuted(x, axis=1, out=x)
    expected = np.array([[0, 9, 3, 6], [14, 20, 11, 17]])
    assert_array_equal(y, expected)
    x1[:, ::3] = expected
    assert_array_equal(x1, x0)