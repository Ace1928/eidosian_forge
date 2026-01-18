import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_shuffle_custom_axis_empty(self):
    random = Generator(MT19937(self.seed))
    desired = np.array([]).reshape((0, 6))
    for axis in (0, 1):
        actual = np.array([]).reshape((0, 6))
        random.shuffle(actual, axis=axis)
        assert_array_equal(actual, desired)