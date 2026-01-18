import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_object_iter_cleanup():
    assert_raises(TypeError, lambda: np.zeros((17000, 2), dtype='f4') * None)
    arr = np.arange(np.BUFSIZE * 10).reshape(10, -1).astype(str)
    oarr = arr.astype(object)
    oarr[:, -1] = None
    assert_raises(TypeError, lambda: np.add(oarr[:, ::-1], arr[:, ::-1]))

    class T:

        def __bool__(self):
            raise TypeError('Ambiguous')
    assert_raises(TypeError, np.logical_or.reduce, np.array([T(), T()], dtype='O'))