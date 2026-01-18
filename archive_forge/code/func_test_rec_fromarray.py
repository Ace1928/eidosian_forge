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
def test_rec_fromarray(self):
    x1 = np.array([[1, 2], [3, 4], [5, 6]])
    x2 = np.array(['a', 'dd', 'xyz'])
    x3 = np.array([1.1, 2, 3])
    np.rec.fromarrays([x1, x2, x3], formats='(2,)i4,a3,f8')