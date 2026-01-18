import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_shuffle_axis_nonsquare(self):
    y1 = np.arange(20).reshape(2, 10)
    y2 = y1.copy()
    random = Generator(MT19937(self.seed))
    random.shuffle(y1, axis=1)
    random = Generator(MT19937(self.seed))
    random.shuffle(y2.T)
    assert_array_equal(y1, y2)