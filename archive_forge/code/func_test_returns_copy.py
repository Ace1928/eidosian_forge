import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_returns_copy(self, block):
    a = np.eye(3)
    b = block(a)
    b[0, 0] = 2
    assert b[0, 0] != a[0, 0]