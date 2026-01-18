import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_zerosSD(self):
    """Check creation of single-dimensional objects"""
    h = np.zeros((2,), dtype=self._descr)
    assert_(normalize_descr(self._descr) == h.dtype.descr)
    assert_(h.dtype['y'].name[:4] == 'void')
    assert_(h.dtype['y'].char == 'V')
    assert_(h.dtype['y'].type == np.void)
    assert_equal(h['z'], np.zeros((2,), dtype='u1'))