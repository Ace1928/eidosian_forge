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
def test_single_two_branches_same_cond(self):

    def impl(x):
        if x is None:
            y = 10
        else:
            y = 40
        if x is not None:
            z = 100
        else:
            z = 400
        return (z, y)
    self.assert_prune(impl, (types.NoneType('none'),), [False, True], None)
    self.assert_prune(impl, (types.IntegerLiteral(10),), [True, False], 10)