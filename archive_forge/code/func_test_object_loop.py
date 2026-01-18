import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_object_loop(self):

    class Mult:

        def __mul__(self, other):
            return 42
    objMult = np.array([Mult()])
    objNULL = np.ndarray(buffer=b'\x00' * np.intp(0).itemsize, shape=1, dtype=object)
    with pytest.raises(TypeError):
        np.einsum('i,j', [1], objNULL)
    with pytest.raises(TypeError):
        np.einsum('i,j', objNULL, [1])
    assert np.einsum('i,j', objMult, objMult) == 42