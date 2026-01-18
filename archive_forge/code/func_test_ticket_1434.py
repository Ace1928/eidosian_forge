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
def test_ticket_1434(self):
    data = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    out = np.zeros((3,))
    ret = data.var(axis=1, out=out)
    assert_(ret is out)
    assert_array_equal(ret, data.var(axis=1))
    ret = data.std(axis=1, out=out)
    assert_(ret is out)
    assert_array_equal(ret, data.std(axis=1))