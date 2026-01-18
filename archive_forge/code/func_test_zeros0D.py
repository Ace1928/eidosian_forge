import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_zeros0D(self):
    """Check creation of 0-dimensional objects"""
    h = np.zeros((), dtype=self._descr)
    assert_(normalize_descr(self._descr) == h.dtype.descr)
    assert_(h.dtype.fields['x'][0].name[:4] == 'void')
    assert_(h.dtype.fields['x'][0].char == 'V')
    assert_(h.dtype.fields['x'][0].type == np.void)
    assert_equal(h['z'], np.zeros((), dtype='u1'))