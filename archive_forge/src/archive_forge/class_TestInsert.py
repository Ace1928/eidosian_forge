from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestInsert(MemoryLeakMixin, TestCase):
    """Test list insert. """

    def test_list_insert_empty(self):

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.insert(i, 1)
            return (len(l), l[0])
        for i in (-10, -5, -1, 0, 1, 4, 9):
            self.assertEqual(foo(i), (1, 1))

    def test_list_insert_singleton(self):

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.append(0)
            l.insert(i, 1)
            return (len(l), l[0], l[1])
        for i in (-10, -3, -2, -1, 0):
            self.assertEqual(foo(i), (2, 1, 0))
        for i in (1, 2, 3, 10):
            self.assertEqual(foo(i), (2, 0, 1))

    def test_list_insert_multiple(self):

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in range(10):
                l.append(0)
            l.insert(i, 1)
            return (len(l), l[i])
        for i in (0, 4, 9):
            self.assertEqual(foo(i), (11, 1))

    def test_list_insert_multiple_before(self):

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in range(10):
                l.append(0)
            l.insert(i, 1)
            return (len(l), l[0])
        for i in (-12, -11, -10, 0):
            self.assertEqual(foo(i), (11, 1))

    def test_list_insert_multiple_after(self):

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in range(10):
                l.append(0)
            l.insert(i, 1)
            return (len(l), l[10])
        for i in (10, 11, 12):
            self.assertEqual(foo(i), (11, 1))

    def test_list_insert_typing_error(self):
        self.disable_leak_check()

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.insert('a', 0)
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('list insert indices must be integers', str(raises.exception))