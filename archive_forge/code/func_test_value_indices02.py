import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_value_indices02():
    """Test input checking"""
    data = np.zeros((5, 4), dtype=np.float32)
    msg = "Parameter 'arr' must be an integer array"
    with assert_raises(ValueError, match=msg):
        ndimage.value_indices(data)