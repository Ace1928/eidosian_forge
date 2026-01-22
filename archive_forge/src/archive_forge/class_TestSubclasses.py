import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestSubclasses:

    def test_basic(self):

        class SubClass(np.ndarray):
            pass
        a = np.arange(5)
        s = a.view(SubClass)
        s_slice = s[:3]
        assert_(type(s_slice) is SubClass)
        assert_(s_slice.base is s)
        assert_array_equal(s_slice, a[:3])
        s_fancy = s[[0, 1, 2]]
        assert_(type(s_fancy) is SubClass)
        assert_(s_fancy.base is not s)
        assert_(type(s_fancy.base) is np.ndarray)
        assert_array_equal(s_fancy, a[[0, 1, 2]])
        assert_array_equal(s_fancy.base, a[[0, 1, 2]])
        s_bool = s[s > 0]
        assert_(type(s_bool) is SubClass)
        assert_(s_bool.base is not s)
        assert_(type(s_bool.base) is np.ndarray)
        assert_array_equal(s_bool, a[a > 0])
        assert_array_equal(s_bool.base, a[a > 0])

    def test_fancy_on_read_only(self):

        class SubClass(np.ndarray):
            pass
        a = np.arange(5)
        s = a.view(SubClass)
        s.flags.writeable = False
        s_fancy = s[[0, 1, 2]]
        assert_(s_fancy.flags.writeable)

    def test_finalize_gets_full_info(self):

        class SubClass(np.ndarray):

            def __array_finalize__(self, old):
                self.finalize_status = np.array(self)
                self.old = old
        s = np.arange(10).view(SubClass)
        new_s = s[:3]
        assert_array_equal(new_s.finalize_status, new_s)
        assert_array_equal(new_s.old, s)
        new_s = s[[0, 1, 2, 3]]
        assert_array_equal(new_s.finalize_status, new_s)
        assert_array_equal(new_s.old, s)
        new_s = s[s > 0]
        assert_array_equal(new_s.finalize_status, new_s)
        assert_array_equal(new_s.old, s)