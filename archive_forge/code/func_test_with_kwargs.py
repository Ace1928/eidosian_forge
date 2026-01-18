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