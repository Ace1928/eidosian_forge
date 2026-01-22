import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class TestMaskedView:

    def setup_method(self):
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        data = np.array(iterator)
        a = array(iterator, dtype=[('a', float), ('b', float)])
        a.mask[0] = (1, 0)
        controlmask = np.array([1] + 19 * [0], dtype=bool)
        self.data = (data, a, controlmask)

    def test_view_to_nothing(self):
        data, a, controlmask = self.data
        test = a.view()
        assert_(isinstance(test, MaskedArray))
        assert_equal(test._data, a._data)
        assert_equal(test._mask, a._mask)

    def test_view_to_type(self):
        data, a, controlmask = self.data
        test = a.view(np.ndarray)
        assert_(not isinstance(test, MaskedArray))
        assert_equal(test, a._data)
        assert_equal_records(test, data.view(a.dtype).squeeze())

    def test_view_to_simple_dtype(self):
        data, a, controlmask = self.data
        test = a.view(float)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, data.ravel())
        assert_equal(test.mask, controlmask)

    def test_view_to_flexible_dtype(self):
        data, a, controlmask = self.data
        test = a.view([('A', float), ('B', float)])
        assert_equal(test.mask.dtype.names, ('A', 'B'))
        assert_equal(test['A'], a['a'])
        assert_equal(test['B'], a['b'])
        test = a[0].view([('A', float), ('B', float)])
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.mask.dtype.names, ('A', 'B'))
        assert_equal(test['A'], a['a'][0])
        assert_equal(test['B'], a['b'][0])
        test = a[-1].view([('A', float), ('B', float)])
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.dtype.names, ('A', 'B'))
        assert_equal(test['A'], a['a'][-1])
        assert_equal(test['B'], a['b'][-1])

    def test_view_to_subdtype(self):
        data, a, controlmask = self.data
        test = a.view((float, 2))
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, data)
        assert_equal(test.mask, controlmask.reshape(-1, 2))
        test = a[0].view((float, 2))
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, data[0])
        assert_equal(test.mask, (1, 0))
        test = a[-1].view((float, 2))
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, data[-1])

    def test_view_to_dtype_and_type(self):
        data, a, controlmask = self.data
        test = a.view((float, 2), np.recarray)
        assert_equal(test, data)
        assert_(isinstance(test, np.recarray))
        assert_(not isinstance(test, MaskedArray))