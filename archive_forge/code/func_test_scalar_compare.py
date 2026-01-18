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
def test_scalar_compare(self):
    a = np.array(['test', 'auto'])
    assert_array_equal(a == 'auto', np.array([False, True]))
    assert_(a[1] == 'auto')
    assert_(a[0] != 'auto')
    b = np.linspace(0, 10, 11)
    assert_array_equal(b != 'auto', np.ones(11, dtype=bool))
    assert_(b[0] != 'auto')