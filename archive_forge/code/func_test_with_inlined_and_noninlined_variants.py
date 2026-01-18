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
def test_with_inlined_and_noninlined_variants(self):

    @overload(len, inline='always')
    def overload_len(A):
        if False:
            return lambda A: 10

    def impl():
        return len([2, 3, 4])
    self.check(impl, inline_expect={'len': False})