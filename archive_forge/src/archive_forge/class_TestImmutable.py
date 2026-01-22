from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestImmutable(MemoryLeakMixin, TestCase):

    def test_is_immutable(self):

        @njit
        def foo():
            l = make_test_list()
            return l._is_mutable()
        self.assertTrue(foo())

    def test_make_immutable_is_immutable(self):

        @njit
        def foo():
            l = make_test_list()
            l._make_immutable()
            return l._is_mutable()
        self.assertFalse(foo())

    def test_length_still_works_when_immutable(self):

        @njit
        def foo():
            l = make_test_list()
            l._make_immutable()
            return (len(l), l._is_mutable())
        length, mutable = foo()
        self.assertEqual(length, 1)
        self.assertFalse(mutable)

    def test_getitem_still_works_when_immutable(self):

        @njit
        def foo():
            l = make_test_list()
            l._make_immutable()
            return (l[0], l._is_mutable())
        test_item, mutable = foo()
        self.assertEqual(test_item, 1)
        self.assertFalse(mutable)

    def test_append_fails(self):
        self.disable_leak_check()

        @njit
        def foo():
            l = make_test_list()
            l._make_immutable()
            l.append(int32(1))
        with self.assertRaises(ValueError) as raises:
            foo()
        self.assertIn('list is immutable', str(raises.exception))

    def test_mutation_fails(self):
        """ Test that any attempt to mutate an immutable typed list fails. """
        self.disable_leak_check()

        def generate_function(line):
            context = {}
            exec(dedent('\n                from numba.typed import listobject\n                from numba import int32\n                def bar():\n                    lst = listobject.new_list(int32)\n                    lst.append(int32(1))\n                    lst._make_immutable()\n                    zero = int32(0)\n                    {}\n                '.format(line)), context)
            return njit(context['bar'])
        for line in ('lst.append(zero)', 'lst[0] = zero', 'lst.pop()', 'del lst[0]', 'lst.extend((zero,))', 'lst.insert(0, zero)', 'lst.clear()', 'lst.reverse()', 'lst.sort()'):
            foo = generate_function(line)
            with self.assertRaises(ValueError) as raises:
                foo()
            self.assertIn('list is immutable', str(raises.exception))