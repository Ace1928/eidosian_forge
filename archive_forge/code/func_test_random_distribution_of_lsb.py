import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('dtype, uint_view_type', [(np.float32, np.uint32), (np.float64, np.uint64)])
def test_random_distribution_of_lsb(self, dtype, uint_view_type):
    random = Generator(MT19937(self.seed))
    sample = random.random(100000, dtype=dtype)
    num_ones_in_lsb = np.count_nonzero(sample.view(uint_view_type) & 1)
    assert 24100 < num_ones_in_lsb < 25900