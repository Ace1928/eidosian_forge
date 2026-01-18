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
def test_single_if_else_two_const_val(self):

    def impl(x, y):
        if x == y:
            return 3.14159
        else:
            return 1.61803
    self.assert_prune(impl, (types.IntegerLiteral(100),) * 2, [None], 100, 100)
    self.assert_prune(impl, (types.NoneType('none'),) * 2, [False], None, None)
    self.assert_prune(impl, (types.IntegerLiteral(100), types.NoneType('none')), [True], 100, None)
    self.assert_prune(impl, (types.IntegerLiteral(100), types.IntegerLiteral(1000)), [None], 100, 1000)