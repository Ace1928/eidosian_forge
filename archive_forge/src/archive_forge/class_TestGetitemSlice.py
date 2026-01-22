from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestGetitemSlice(MemoryLeakMixin, TestCase):
    """Test list getitem when indexing with slices. """

    def test_list_getitem_empty_slice_defaults(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            n = l[:]
            return len(n)
        self.assertEqual(foo(), 0)

    def test_list_getitem_singleton_slice_defaults(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            n = l[:]
            return len(n)
        self.assertEqual(foo(), 1)

    def test_list_getitem_multiple_slice_defaults(self):

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[:]
            return n[i]
        for i, j in ((0, 10), (9, 19), (4, 14), (-5, 15), (-1, 19), (-10, 10)):
            self.assertEqual(foo(i), j)

    def test_list_getitem_multiple_slice_pos_start(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[5:]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (15, 16, 17, 18, 19))

    def test_list_getitem_multiple_slice_pos_stop(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[:5]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (10, 11, 12, 13, 14))

    def test_list_getitem_multiple_slice_pos_start_pos_stop(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[2:7]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (12, 13, 14, 15, 16))

    def test_list_getitem_multiple_slice_pos_start_pos_stop_pos_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[1:9:2]
            return (len(n), (n[0], n[1], n[2], n[3]))
        length, items = foo()
        self.assertEqual(length, 4)
        self.assertEqual(items, (11, 13, 15, 17))

    def test_list_getitem_multiple_slice_neg_start(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[-5:]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (15, 16, 17, 18, 19))

    def test_list_getitem_multiple_slice_neg_stop(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[:-5]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (10, 11, 12, 13, 14))

    def test_list_getitem_multiple_slice_neg_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[::-2]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (19, 17, 15, 13, 11))

    def test_list_getitem_multiple_slice_pos_start_neg_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[4::-1]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (14, 13, 12, 11, 10))

    def test_list_getitem_multiple_slice_neg_start_neg_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[-6::-1]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (14, 13, 12, 11, 10))

    def test_list_getitem_multiple_slice_pos_stop_neg_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[:4:-1]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (19, 18, 17, 16, 15))

    def test_list_getitem_multiple_slice_neg_stop_neg_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[:-6:-1]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (19, 18, 17, 16, 15))

    def test_list_getitem_multiple_slice_pos_start_pos_stop_neg_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[8:3:-1]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (18, 17, 16, 15, 14))

    def test_list_getitem_multiple_slice_neg_start_neg_stop_neg_step(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[-2:-7:-1]
            return (len(n), (n[0], n[1], n[2], n[3], n[4]))
        length, items = foo()
        self.assertEqual(length, 5)
        self.assertEqual(items, (18, 17, 16, 15, 14))

    def test_list_getitem_multiple_slice_start_out_of_range(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[10:]
            return len(n)
        self.assertEqual(foo(), 0)

    def test_list_getitem_multiple_slice_stop_zero(self):

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            n = l[:0]
            return len(n)
        self.assertEqual(foo(), 0)

    def test_list_getitem_multiple_slice_zero_step_index_error(self):
        self.disable_leak_check()

        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            l[::0]
        with self.assertRaises(ValueError) as raises:
            foo()
        self.assertIn('slice step cannot be zero', str(raises.exception))