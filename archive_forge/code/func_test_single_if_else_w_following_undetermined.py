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
def test_single_if_else_w_following_undetermined(self):

    def impl(x):
        x_is_none_work = False
        if x is None:
            x_is_none_work = True
        else:
            dead = 7
        if x_is_none_work:
            y = 10
        else:
            y = -3
        return y
    self.assert_prune(impl, (types.NoneType('none'),), [False, None], None)
    self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)

    def impl(x):
        x_is_none_work = False
        if x is None:
            x_is_none_work = True
        else:
            pass
        if x_is_none_work:
            y = 10
        else:
            y = -3
        return y
    if utils.PYVERSION >= (3, 10):
        self.assert_prune(impl, (types.NoneType('none'),), [False, None], None)
    else:
        self.assert_prune(impl, (types.NoneType('none'),), [None, None], None)
    self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)