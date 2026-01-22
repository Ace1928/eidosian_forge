import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestMultipleFields:

    def setup_method(self):
        self.ary = np.array([(1, 2, 3, 4), (5, 6, 7, 8)], dtype='i4,f4,i2,c8')

    def _bad_call(self):
        return self.ary['f0', 'f1']

    def test_no_tuple(self):
        assert_raises(IndexError, self._bad_call)

    def test_return(self):
        res = self.ary[['f0', 'f2']].tolist()
        assert_(res == [(1, 3), (5, 7)])