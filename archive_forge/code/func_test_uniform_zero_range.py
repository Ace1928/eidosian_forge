import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_uniform_zero_range(self):
    func = random.uniform
    result = func(1.5, 1.5)
    assert_allclose(result, 1.5)
    result = func([0.0, np.pi], [0.0, np.pi])
    assert_allclose(result, [0.0, np.pi])
    result = func([[2145.12], [2145.12]], [2145.12, 2145.12])
    assert_allclose(result, 2145.12 + np.zeros((2, 2)))