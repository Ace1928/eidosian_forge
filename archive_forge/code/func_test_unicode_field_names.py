import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_unicode_field_names(tmpdir):
    arr = np.array([(1, 3), (1, 2), (1, 3), (1, 2)], dtype=[('int', int), ('整形', int)])
    fname = os.path.join(tmpdir, 'unicode.npy')
    with open(fname, 'wb') as f:
        format.write_array(f, arr, version=(3, 0))
    with open(fname, 'rb') as f:
        arr2 = format.read_array(f)
    assert_array_equal(arr, arr2)
    with open(fname, 'wb') as f:
        with assert_warns(UserWarning):
            format.write_array(f, arr, version=None)