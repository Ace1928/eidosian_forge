import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_object_iter_cleanup_reduce():
    arr = np.array([[None, 1], [-1, -1], [None, 2], [-1, -1]])[::2]
    with pytest.raises(TypeError):
        np.sum(arr)