import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_shuffle_exceptions(self):
    random = Generator(MT19937(self.seed))
    arr = np.arange(10)
    assert_raises(np.AxisError, random.shuffle, arr, 1)
    arr = np.arange(9).reshape((3, 3))
    assert_raises(np.AxisError, random.shuffle, arr, 3)
    assert_raises(TypeError, random.shuffle, arr, slice(1, 2, None))
    arr = [[1, 2, 3], [4, 5, 6]]
    assert_raises(NotImplementedError, random.shuffle, arr, 1)
    arr = np.array(3)
    assert_raises(TypeError, random.shuffle, arr)
    arr = np.ones((3, 2))
    assert_raises(np.AxisError, random.shuffle, arr, 2)