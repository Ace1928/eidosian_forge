import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_value_indices03():
    """Test different input array shapes, from 1-D to 4-D"""
    for shape in [(36,), (18, 2), (3, 3, 4), (3, 3, 2, 2)]:
        a = np.array(12 * [1] + 12 * [2] + 12 * [3], dtype=np.int32).reshape(shape)
        trueKeys = np.unique(a)
        vi = ndimage.value_indices(a)
        assert_equal(list(vi.keys()), list(trueKeys))
        for k in trueKeys:
            trueNdx = np.where(a == k)
            assert_equal(vi[k], trueNdx)