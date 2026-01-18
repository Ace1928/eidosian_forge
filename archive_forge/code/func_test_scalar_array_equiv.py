import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_scalar_array_equiv(self, endpoint):
    for dt in self.itype:
        lbnd = 0 if dt is bool else np.iinfo(dt).min
        ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
        ubnd = ubnd - 1 if endpoint else ubnd
        size = 1000
        random = Generator(MT19937(1234))
        scalar = random.integers(lbnd, ubnd, size=size, endpoint=endpoint, dtype=dt)
        random = Generator(MT19937(1234))
        scalar_array = random.integers([lbnd], [ubnd], size=size, endpoint=endpoint, dtype=dt)
        random = Generator(MT19937(1234))
        array = random.integers([lbnd] * size, [ubnd] * size, size=size, endpoint=endpoint, dtype=dt)
        assert_array_equal(scalar, scalar_array)
        assert_array_equal(scalar, array)