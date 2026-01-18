import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_small_boolean_arrays(self):
    a = np.zeros((16, 1, 1), dtype=np.bool_)[:2]
    a[...] = True
    out = np.zeros((16, 1, 1), dtype=np.bool_)[:2]
    tgt = np.ones((2, 1, 1), dtype=np.bool_)
    res = np.einsum('...ij,...jk->...ik', a, a, out=out)
    assert_equal(res, tgt)