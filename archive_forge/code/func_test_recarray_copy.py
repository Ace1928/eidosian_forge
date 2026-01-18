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
def test_recarray_copy(self):
    dt = [('x', np.int16), ('y', np.float64)]
    ra = np.array([(1, 2.3)], dtype=dt)
    rb = np.rec.array(ra, dtype=dt)
    rb['x'] = 2.0
    assert_(ra['x'] != rb['x'])