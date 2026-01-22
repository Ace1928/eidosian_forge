from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
class TestTags:
    """Test that sav files with description tag read at all"""

    def test_description(self):
        s = readsav(path.join(DATA_PATH, 'scalar_byte_descr.sav'), verbose=False)
        assert_identical(s.i8u, np.uint8(234))