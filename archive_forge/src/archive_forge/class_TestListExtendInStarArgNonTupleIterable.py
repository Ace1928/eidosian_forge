import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
class TestListExtendInStarArgNonTupleIterable(MemoryLeakMixin, TestCase):
    """Test `fn(pos_arg0, pos_arg1, *args)` where args is a non-tuple iterable.

    Python 3.9+ will generate LIST_EXTEND bytecode to combine the positional
    arguments with the `*args`.

    See #8059

    NOTE: At the moment, there are no meaningful tests for NoPython because the
    lack of support for `tuple(iterable)` for most iterable types.
    """

    def test_list_extend_forceobj(self):

        def consumer(*x):
            return x

        @jit(forceobj=True)
        def foo(x):
            return consumer(1, 2, *x)
        got = foo('ijo')
        expect = foo.py_func('ijo')
        self.assertEqual(got, (1, 2, 'i', 'j', 'o'))
        self.assertEqual(got, expect)