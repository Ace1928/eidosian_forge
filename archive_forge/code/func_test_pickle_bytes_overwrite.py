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
def test_pickle_bytes_overwrite(self):
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        data = np.array([1], dtype='b')
        data = pickle.loads(pickle.dumps(data, protocol=proto))
        data[0] = 125
        bytestring = '\x01  '.encode('ascii')
        assert_equal(bytestring[0:1], '\x01'.encode('ascii'))