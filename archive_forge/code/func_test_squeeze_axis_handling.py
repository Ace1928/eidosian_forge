import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_squeeze_axis_handling(self):

    class OldSqueeze(np.ndarray):

        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls)
            return obj

        def squeeze(self):
            return super().squeeze()
    oldsqueeze = OldSqueeze(np.array([[1], [2], [3]]))
    assert_equal(np.squeeze(oldsqueeze), np.array([1, 2, 3]))
    assert_equal(np.squeeze(oldsqueeze, axis=None), np.array([1, 2, 3]))
    with assert_raises(TypeError):
        np.squeeze(oldsqueeze, axis=1)
    with assert_raises(TypeError):
        np.squeeze(oldsqueeze, axis=0)
    with assert_raises(ValueError):
        np.squeeze(np.array([[1], [2], [3]]), axis=0)