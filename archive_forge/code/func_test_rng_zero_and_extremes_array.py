import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_rng_zero_and_extremes_array(self, endpoint):
    size = 1000
    for dt in self.itype:
        lbnd = 0 if dt is bool else np.iinfo(dt).min
        ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
        ubnd = ubnd - 1 if endpoint else ubnd
        tgt = ubnd - 1
        assert_equal(self.rfunc([tgt], [tgt + 1], size=size, dtype=dt), tgt)
        assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
        assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)
        tgt = lbnd
        assert_equal(self.rfunc([tgt], [tgt + 1], size=size, dtype=dt), tgt)
        assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
        assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)
        tgt = (lbnd + ubnd) // 2
        assert_equal(self.rfunc([tgt], [tgt + 1], size=size, dtype=dt), tgt)
        assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
        assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)