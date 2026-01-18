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
def test_object_array_nested(self):
    a = np.array(0, dtype=object)
    b = np.array(0, dtype=object)
    a[()] = b
    assert_equal(int(a), int(0))
    assert_equal(float(a), float(0))