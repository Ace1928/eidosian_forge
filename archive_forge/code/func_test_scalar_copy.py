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
def test_scalar_copy(self):
    scalar_types = set(np.sctypeDict.values())
    values = {np.void: b'a', np.bytes_: b'a', np.str_: 'a', np.datetime64: '2017-08-25'}
    for sctype in scalar_types:
        item = sctype(values.get(sctype, 1))
        item2 = copy.copy(item)
        assert_equal(item, item2)