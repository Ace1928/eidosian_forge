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
def test_object_array_self_copy(self):
    a = np.array(object(), dtype=object)
    np.copyto(a, a)
    if HAS_REFCOUNT:
        assert_(sys.getrefcount(a[()]) == 2)
    a[()].__class__