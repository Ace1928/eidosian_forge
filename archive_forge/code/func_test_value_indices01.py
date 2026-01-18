import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_value_indices01():
    """Test dictionary keys and entries"""
    data = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 2, 2, 0, 0], [0, 0, 2, 2, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 4, 0]])
    vi = ndimage.value_indices(data, ignore_value=0)
    true_keys = [1, 2, 4]
    assert_equal(list(vi.keys()), true_keys)
    truevi = {}
    for k in true_keys:
        truevi[k] = np.where(data == k)
    vi = ndimage.value_indices(data, ignore_value=0)
    assert_equal(vi, truevi)