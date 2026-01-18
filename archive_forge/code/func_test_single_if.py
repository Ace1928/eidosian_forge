import collections
import types as pytypes
import numpy as np
from numba.core.compiler import run_frontend, Flags, StateDict
from numba import jit, njit, literal_unroll
from numba.core import types, errors, ir, rewrites, ir_utils, utils, cpu
from numba.core import postproc
from numba.core.inline_closurecall import InlineClosureCallPass
from numba.tests.support import (TestCase, MemoryLeakMixin, SerialMixin,
from numba.core.analysis import dead_branch_prune, rewrite_semantic_constants
from numba.core.untyped_passes import (ReconstructSSA, TranslateByteCode,
from numba.core.compiler import DefaultPassBuilder, CompilerBase, PassManager
def test_single_if(self):

    def impl(x):
        if 1 == 0:
            return 3.14159
    self.assert_prune(impl, (types.NoneType('none'),), [True], None)

    def impl(x):
        if 1 == 1:
            return 3.14159
    self.assert_prune(impl, (types.NoneType('none'),), [False], None)

    def impl(x):
        if x is None:
            return 3.14159
    self.assert_prune(impl, (types.NoneType('none'),), [False], None)
    self.assert_prune(impl, (types.IntegerLiteral(10),), [True], 10)

    def impl(x):
        if x == 10:
            return 3.14159
    self.assert_prune(impl, (types.NoneType('none'),), [True], None)
    self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)

    def impl(x):
        if x == 10:
            z = 3.14159
    self.assert_prune(impl, (types.NoneType('none'),), [True], None)
    self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)

    def impl(x):
        z = None
        y = z
        if x == y:
            return 100
    self.assert_prune(impl, (types.NoneType('none'),), [False], None)
    self.assert_prune(impl, (types.IntegerLiteral(10),), [True], 10)