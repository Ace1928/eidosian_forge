import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_pad_with_zero(self):
    a = np.ones((3, 5))
    b = np.pad(a, (0, 5), mode='wrap')
    assert_array_equal(a, b[:-5, :-5])