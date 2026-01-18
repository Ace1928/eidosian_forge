import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_repr(self):
    a = np.array([(1, 0.1), (2, 0.2)], dtype=[('foo', '<i4'), ('bar', '<f8')])
    a = np.rec.array(a)
    assert_equal(repr(a), textwrap.dedent("            rec.array([(1, 0.1), (2, 0.2)],\n                      dtype=[('foo', '<i4'), ('bar', '<f8')])"))
    a = np.array(np.ones(4, dtype='f8'))
    assert_(repr(np.rec.array(a)).startswith('rec.array'))
    a = np.rec.array(np.ones(3, dtype='i4,i4'))
    assert_equal(repr(a).find('numpy.record'), -1)
    a = np.rec.array(np.ones(3, dtype='i4'))
    assert_(repr(a).find('dtype=int32') != -1)