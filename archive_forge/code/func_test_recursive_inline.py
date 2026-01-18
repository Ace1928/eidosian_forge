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
@unittest.skip('Need to work out how to prevent this')
def test_recursive_inline(self):

    @njit(inline='always')
    def foo(x):
        if x == 0:
            return 12
        else:
            foo(x - 1)
    a = 3

    def impl():
        b = 0
        if a > 1:
            b += 1
        foo(5)
        if b < a:
            b -= 1
    self.check(impl, inline_expect={'foo': True})