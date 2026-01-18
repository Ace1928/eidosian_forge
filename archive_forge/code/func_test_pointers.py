from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_pointers(self):
    s = readsav(path.join(DATA_PATH, 'scalar_heap_pointer.sav'), verbose=False)
    assert_identical(s.c64_pointer1, np.complex128(1.1987253647623157e+112 - 5.198725888772916e+307j))
    assert_identical(s.c64_pointer2, np.complex128(1.1987253647623157e+112 - 5.198725888772916e+307j))
    assert_(s.c64_pointer1 is s.c64_pointer2)