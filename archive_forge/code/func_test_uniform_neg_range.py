import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_uniform_neg_range(self):
    func = random.uniform
    assert_raises(ValueError, func, 2, 1)
    assert_raises(ValueError, func, [1, 2], [1, 1])
    assert_raises(ValueError, func, [[0, 1], [2, 3]], 2)