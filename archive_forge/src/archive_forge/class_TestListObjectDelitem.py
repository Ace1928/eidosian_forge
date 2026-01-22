from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestListObjectDelitem(MemoryLeakMixin, TestCase):
    """Test list delitem.
    """

    def test_list_singleton_delitem_index(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            del l[0]
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_singleton_delitem_slice_defaults(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            del l[:]
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_singleton_delitem_slice_start(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            del l[0:]
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_singleton_delitem_slice_stop(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            del l[:1]
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_singleton_delitem_slice_start_stop(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            del l[0:1]
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_singleton_delitem_slice_start_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            del l[0::1]
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_singleton_delitem_slice_start_stop_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            del l[0:1:1]
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_multiple_delitem(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in (10, 11, 12):
                l.append(j)
            del l[0]
            return (len(l), l[0], l[1])
        self.assertEqual(foo(), (2, 11, 12))

    def test_list_multiple_delitem_slice(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in (10, 11, 12):
                l.append(j)
            del l[:]
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_multiple_delitem_off_by_one(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            k = listobject.new_list(int32)
            for j in range(10, 20):
                k.append(j)
            del l[-9:-20]
            return k == l
        self.assertTrue(foo())