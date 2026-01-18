import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('mode', ['edge', 'symmetric', 'reflect', 'wrap'])
def test_object_input(mode):
    a = np.full((4, 3), fill_value=None)
    pad_amt = ((2, 3), (3, 2))
    b = np.full((9, 8), fill_value=None)
    assert_array_equal(np.pad(a, pad_amt, mode=mode), b)