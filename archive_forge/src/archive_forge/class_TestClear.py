from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestClear(MemoryLeakMixin, TestCase):
    """Test list clear. """

    def test_list_clear_empty(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.clear()
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_clear_singleton(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            l.clear()
            return len(l)
        self.assertEqual(foo(), 0)

    def test_list_clear_multiple(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10):
                l.append(0)
            l.clear()
            return len(l)
        self.assertEqual(foo(), 0)