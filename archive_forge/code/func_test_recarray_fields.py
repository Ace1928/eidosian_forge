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
def test_recarray_fields(self):
    dt0 = np.dtype([('f0', 'i4'), ('f1', 'i4')])
    dt1 = np.dtype([('f0', 'i8'), ('f1', 'i8')])
    for a in [np.array([(1, 2), (3, 4)], 'i4,i4'), np.rec.array([(1, 2), (3, 4)], 'i4,i4'), np.rec.array([(1, 2), (3, 4)]), np.rec.fromarrays([(1, 2), (3, 4)], 'i4,i4'), np.rec.fromarrays([(1, 2), (3, 4)])]:
        assert_(a.dtype in [dt0, dt1])