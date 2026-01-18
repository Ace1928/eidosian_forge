import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('outshape', [(2, 3), 5])
def test_permuted_out_with_wrong_shape(self, outshape):
    a = np.array([1, 2, 3])
    out = np.zeros(outshape, dtype=a.dtype)
    with pytest.raises(ValueError, match='same shape'):
        random.permuted(a, out=out)