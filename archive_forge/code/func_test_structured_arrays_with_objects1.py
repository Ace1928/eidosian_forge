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
def test_structured_arrays_with_objects1(self):
    stra = 'aaaa'
    strb = 'bbbb'
    x = np.array([[(0, stra), (1, strb)]], 'i8,O')
    x[x.nonzero()] = x.ravel()[:1]
    assert_(x[0, 1] == x[0, 0])