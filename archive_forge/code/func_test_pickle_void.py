import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_pickle_void(self):
    dt = np.dtype([('obj', 'O'), ('int', 'i')])
    a = np.empty(1, dtype=dt)
    data = (bytearray(b'eman'),)
    a['obj'] = data
    a['int'] = 42
    ctor, args = a[0].__reduce__()
    assert ctor is np.core.multiarray.scalar
    dtype, obj = args
    assert not isinstance(obj, bytes)
    assert_raises(RuntimeError, ctor, dtype, 13)
    dump = pickle.dumps(a[0])
    unpickled = pickle.loads(dump)
    assert a[0] == unpickled
    with pytest.warns(DeprecationWarning):
        assert ctor(np.dtype('O'), data) is data