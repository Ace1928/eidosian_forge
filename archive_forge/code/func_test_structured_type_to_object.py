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
def test_structured_type_to_object(self):
    a_rec = np.array([(0, 1), (3, 2)], dtype='i4,i8')
    a_obj = np.empty((2,), dtype=object)
    a_obj[0] = (0, 1)
    a_obj[1] = (3, 2)
    assert_equal(a_rec.astype(object), a_obj)
    b = np.empty_like(a_obj)
    b[...] = a_rec
    assert_equal(b, a_obj)
    b = np.empty_like(a_rec)
    b[...] = a_obj
    assert_equal(b, a_rec)