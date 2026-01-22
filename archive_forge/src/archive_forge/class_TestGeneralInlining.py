import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
class TestGeneralInlining(MemoryLeakMixin, InliningBase):

    def test_with_inlined_and_noninlined_variants(self):

        @overload(len, inline='always')
        def overload_len(A):
            if False:
                return lambda A: 10

        def impl():
            return len([2, 3, 4])
        self.check(impl, inline_expect={'len': False})

    def test_with_kwargs(self):

        def foo(a, b=3, c=5):
            return a + b + c

        @overload(foo, inline='always')
        def overload_foo(a, b=3, c=5):

            def impl(a, b=3, c=5):
                return a + b + c
            return impl

        def impl():
            return foo(3, c=10)
        self.check(impl, inline_expect={'foo': True})

    def test_with_kwargs2(self):

        @njit(inline='always')
        def bar(a, b=12, c=9):
            return a + b

        def impl(a, b=7, c=5):
            return bar(a + b, c=19)
        self.check(impl, 3, 4, inline_expect={'bar': True})

    def test_inlining_optional_constant(self):

        @njit(inline='always')
        def bar(a=None, b=None):
            if b is None:
                b = 123
            return (a, b)

        def impl():
            return (bar(), bar(123), bar(b=321))
        self.check(impl, block_count='SKIP', inline_expect={'bar': True})